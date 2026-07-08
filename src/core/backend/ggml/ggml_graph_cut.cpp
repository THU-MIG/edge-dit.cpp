#include "backend/ggml/ggml_graph_cut.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "utils/util.h"
#include "backend/ggml/parallel/ggml_comm.hpp"

namespace sd::ggml_graph_cut {

    namespace {
        thread_local std::vector<CommMark> g_comm_marks;
    }

    const char* comm_kind_name(Segment::CommKind kind);

    static std::string graph_cut_tensor_display_name(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return "<null>";
        }
        if (tensor->name[0] != '\0') {
            return tensor->name;
        }
        return sd_format("<tensor@%p>", (const void*)tensor);
    }

    static int graph_leaf_index(ggml_cgraph* gf, const ggml_tensor* tensor) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(tensor != nullptr);
        for (int i = 0; i < gf->n_leafs; ++i) {
            if (gf->leafs[i] == tensor) {
                return i;
            }
        }
        return -1;
    }

    static bool is_params_tensor(const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                                 const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return false;
        }
        return params_tensor_set.find(tensor) != params_tensor_set.end();
    }

    static Plan::InputShape input_shape(const ggml_tensor* tensor) {
        Plan::InputShape shape;
        if (tensor == nullptr) {
            return shape;
        }
        shape.type = tensor->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            shape.ne[static_cast<size_t>(i)] = tensor->ne[i];
        }
        return shape;
    }

    static size_t graph_cut_segment_vram_bytes(const Segment& segment) {
        return segment.compute_buffer_size +
               segment.input_param_bytes +
               segment.input_previous_cut_bytes +
               segment.output_bytes;
    }

    static void rebuild_segment_graph_usage(ggml_cgraph* graph) {
        GGML_ASSERT(graph != nullptr);

        ggml_hash_set_reset(&graph->visited_hash_set);
        std::memset(graph->use_counts,
                    0,
                    graph->visited_hash_set.size * sizeof(graph->use_counts[0]));

        auto register_tensor = [graph](ggml_tensor* tensor) {
            if (tensor == nullptr) {
                return;
            }
            ggml_hash_find_or_insert(&graph->visited_hash_set, tensor);
        };

        for (int i = 0; i < graph->n_leafs; ++i) {
            register_tensor(graph->leafs[i]);
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            register_tensor(graph->nodes[i]);
        }

        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor* node = graph->nodes[i];
            if (node == nullptr) {
                continue;
            }
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ggml_tensor* src = node->src[src_idx];
                if (src == nullptr) {
                    continue;
                }
                const size_t src_hash_pos =
                    ggml_hash_find_or_insert(&graph->visited_hash_set, src);
                graph->use_counts[src_hash_pos]++;
            }
        }
    }

    static void populate_future_input_names(Plan& plan) {
        std::unordered_set<std::string> future_input_names;
        for (auto it = plan.segments.rbegin(); it != plan.segments.rend(); ++it) {
            it->future_input_names = future_input_names;
            for (const auto& input_ref : it->input_refs) {
                if (input_ref.type == Segment::INPUT_PREVIOUS_CUT &&
                    !input_ref.display_name.empty()) {
                    future_input_names.insert(input_ref.display_name);
                }
            }
        }
    }

    static Segment make_segment_seed(const Plan& plan,
                                     size_t start_segment_index,
                                     size_t end_segment_index) {
        GGML_ASSERT(start_segment_index < plan.segments.size());
        GGML_ASSERT(end_segment_index < plan.segments.size());
        GGML_ASSERT(start_segment_index <= end_segment_index);

        Segment seed;
        const auto& start_segment  = plan.segments[start_segment_index];
        const auto& target_segment = plan.segments[end_segment_index];
        std::unordered_set<int> seen_output_node_indices;
        std::unordered_set<int> seen_seed_node_indices;
        for (size_t seg_idx = start_segment_index; seg_idx <= end_segment_index; ++seg_idx) {
            for (int output_node_index : plan.segments[seg_idx].output_node_indices) {
                if (seen_output_node_indices.insert(output_node_index).second) {
                    seed.output_node_indices.push_back(output_node_index);
                }
            }
            for (int seed_node_index : plan.segments[seg_idx].seed_node_indices) {
                if (seen_seed_node_indices.insert(seed_node_index).second) {
                    seed.seed_node_indices.push_back(seed_node_index);
                }
            }
        }
        if (start_segment_index == end_segment_index) {
            seed.group_name = target_segment.group_name;
        } else {
            seed.group_name = sd_format("%s..%s",
                                        start_segment.group_name.c_str(),
                                        target_segment.group_name.c_str());
        }
        return seed;
    }

    static void build_segment(ggml_cgraph* gf,
                              Plan& plan,
                              Segment& segment,
                              const std::unordered_map<const ggml_tensor*, int>& producer_index,
                              std::unordered_set<int>& available_cut_output_node_indices,
                              ggml_backend_t backend,
                              const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                              const char* log_desc) {
        std::set<int> internal_nodes;
        std::unordered_set<const ggml_tensor*> input_seen;
        std::vector<Segment::InputRef> input_refs;

        std::stack<ggml_tensor*> work_stack;
        for (int output_node_index : segment.output_node_indices) {
            ggml_tensor* output = ggml_graph_node(gf, output_node_index);
            if (output != nullptr) {
                work_stack.push(output);
            }
        }
        for (int seed_node_index : segment.seed_node_indices) {
            if (seed_node_index < 0 || seed_node_index >= ggml_graph_n_nodes(gf)) {
                continue;
            }
            ggml_tensor* seed = ggml_graph_node(gf, seed_node_index);
            if (seed != nullptr) {
                work_stack.push(seed);
            }
        }

        while (!work_stack.empty()) {
            ggml_tensor* tensor = work_stack.top();
            work_stack.pop();

            if (tensor == nullptr) {
                continue;
            }

            auto producer_it = producer_index.find(tensor);
            if (producer_it == producer_index.end()) {
                if (input_seen.insert(tensor).second) {
                    Segment::InputRef input_ref;
                    input_ref.type         = is_params_tensor(params_tensor_set, tensor) ? Segment::INPUT_PARAM : Segment::INPUT_EXTERNAL;
                    input_ref.display_name = graph_cut_tensor_display_name(tensor);
                    input_ref.leaf_index   = graph_leaf_index(gf, tensor);
                    input_refs.push_back(std::move(input_ref));
                }
                continue;
            }

            int node_idx = producer_it->second;
            if (available_cut_output_node_indices.find(node_idx) != available_cut_output_node_indices.end()) {
                if (input_seen.insert(tensor).second) {
                    Segment::InputRef input_ref;
                    input_ref.type         = Segment::INPUT_PREVIOUS_CUT;
                    input_ref.display_name = graph_cut_tensor_display_name(tensor);
                    input_ref.node_index   = node_idx;
                    input_refs.push_back(std::move(input_ref));
                }
                continue;
            }

            if (!internal_nodes.insert(node_idx).second) {
                continue;
            }

            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node->view_src != nullptr) {
                work_stack.push(node->view_src);
            }
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                if (node->src[src_idx] != nullptr) {
                    work_stack.push(node->src[src_idx]);
                }
            }
        }

        if (!internal_nodes.empty()) {
            segment.internal_node_indices.assign(internal_nodes.begin(), internal_nodes.end());
        }

        std::sort(input_refs.begin(),
                  input_refs.end(),
                  [](const Segment::InputRef& a, const Segment::InputRef& b) {
                      if (a.type != b.type) {
                          return a.type < b.type;
                      }
                      return a.display_name < b.display_name;
                  });
        segment.input_refs = input_refs;
        for (const auto& input : input_refs) {
            ggml_tensor* current_input = input_tensor(gf, input);
            size_t tensor_bytes        = current_input == nullptr
                                             ? 0
                                             : (input.type == Segment::INPUT_PREVIOUS_CUT
                                                    ? cache_tensor_bytes(current_input)
                                                    : ggml_nbytes(current_input));
            switch (input.type) {
                case Segment::INPUT_PREVIOUS_CUT:
                    segment.input_previous_cut_bytes += tensor_bytes;
                    break;
                case Segment::INPUT_PARAM:
                    segment.input_param_bytes += tensor_bytes;
                    break;
                case Segment::INPUT_EXTERNAL:
                default:
                    segment.input_external_bytes += tensor_bytes;
                    break;
            }
        }
        for (int output_node_index : segment.output_node_indices) {
            ggml_tensor* output = ggml_graph_node(gf, output_node_index);
            segment.output_bytes += cache_tensor_bytes(output);
        }
        segment.compute_buffer_size = measure_segment_compute_buffer(backend, gf, segment, log_desc);

        for (int output_node_index : segment.output_node_indices) {
            available_cut_output_node_indices.insert(output_node_index);
        }
        plan.segments.push_back(std::move(segment));
    }

    static std::vector<Segment> topo_sort_segments(ggml_cgraph* gf,
                                                   const std::vector<Segment>& segments,
                                                   const std::unordered_map<const ggml_tensor*, int>& producer_index,
                                                   const char* log_desc) {
        if (segments.size() <= 1) {
            return segments;
        }

        std::unordered_map<int, size_t> output_node_to_segment;
        output_node_to_segment.reserve(segments.size());
        for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
            for (int output_node_index : segments[seg_idx].output_node_indices) {
                output_node_to_segment[output_node_index] = seg_idx;
            }
        }

        std::vector<std::unordered_set<size_t>> dependencies(segments.size());
        std::vector<std::vector<size_t>> dependents(segments.size());
        for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
            std::stack<int> work_stack;
            std::unordered_set<int> visited_node_indices;
            for (int output_node_index : segments[seg_idx].output_node_indices) {
                work_stack.push(output_node_index);
            }
            for (int seed_node_index : segments[seg_idx].seed_node_indices) {
                work_stack.push(seed_node_index);
            }

            while (!work_stack.empty()) {
                const int node_idx = work_stack.top();
                work_stack.pop();
                if (node_idx < 0 || node_idx >= ggml_graph_n_nodes(gf)) {
                    continue;
                }
                if (!visited_node_indices.insert(node_idx).second) {
                    continue;
                }

                auto cut_segment_it = output_node_to_segment.find(node_idx);
                if (cut_segment_it != output_node_to_segment.end() &&
                    cut_segment_it->second != seg_idx) {
                    dependencies[seg_idx].insert(cut_segment_it->second);
                    continue;
                }

                ggml_tensor* node = ggml_graph_node(gf, node_idx);
                if (node == nullptr) {
                    continue;
                }
                auto push_producer = [&](ggml_tensor* tensor) {
                    if (tensor == nullptr) {
                        return;
                    }
                    auto producer_it = producer_index.find(tensor);
                    if (producer_it != producer_index.end()) {
                        work_stack.push(producer_it->second);
                    }
                };
                push_producer(node->view_src);
                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    push_producer(node->src[src_idx]);
                }
            }
        }

        std::vector<size_t> indegree(segments.size(), 0);
        for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
            indegree[seg_idx] = dependencies[seg_idx].size();
            for (size_t dependency : dependencies[seg_idx]) {
                dependents[dependency].push_back(seg_idx);
            }
        }

        std::vector<Segment> sorted_segments;
        sorted_segments.reserve(segments.size());
        std::vector<bool> emitted(segments.size(), false);
        for (size_t emitted_count = 0; emitted_count < segments.size(); ++emitted_count) {
            size_t ready = segments.size();
            for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
                if (!emitted[seg_idx] && indegree[seg_idx] == 0) {
                    ready = seg_idx;
                    break;
                }
            }

            if (ready == segments.size()) {
                if (log_desc != nullptr) {
                    LOG_WARN("%s graph cut segment dependency cycle detected; keeping remaining segment order",
                             log_desc);
                }
                for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
                    if (!emitted[seg_idx]) {
                        sorted_segments.push_back(segments[seg_idx]);
                    }
                }
                return sorted_segments;
            }

            emitted[ready] = true;
            sorted_segments.push_back(segments[ready]);
            for (size_t dependent : dependents[ready]) {
                GGML_ASSERT(indegree[dependent] > 0);
                --indegree[dependent];
            }
        }

        return sorted_segments;
    }

    bool is_graph_cut_tensor(const ggml_tensor* tensor) {
        if (tensor == nullptr || tensor->name[0] == '\0') {
            return false;
        }
        return std::strncmp(tensor->name, GGML_RUNNER_CUT_PREFIX, std::strlen(GGML_RUNNER_CUT_PREFIX)) == 0;
    }

    std::string make_graph_cut_name(const std::string& group, const std::string& output) {
        return std::string(GGML_RUNNER_CUT_PREFIX) + group + "|" + output;
    }

    void mark_graph_cut(ggml_tensor* tensor, const std::string& group, const std::string& output) {
        if (tensor == nullptr) {
            return;
        }
        auto name = make_graph_cut_name(group, output);
        ggml_set_name(tensor, name.c_str());
    }

    void mark_comm_op(ggml_tensor* input,
                      ggml_tensor* output,
                      Segment::CommKind kind,
                      const std::string& name,
                      edgedit::parallel::ReduceOp reduce_op,
                      size_t count_per_peer,
                      int root) {
        if (input == nullptr || kind == Segment::CommKind::NONE) {
            return;
        }
        if (kind != Segment::CommKind::BROADCAST && output == nullptr) {
            return;
        }

        CommMark mark;
        mark.kind = kind;
        mark.input = input;
        mark.output = output;
        mark.name = name;
        mark.reduce_op = reduce_op;
        mark.count_per_peer = count_per_peer;
        mark.root = root;
        g_comm_marks.push_back(std::move(mark));
    }

    void clear_comm_marks() {
        g_comm_marks.clear();
    }

    static int node_index_for_tensor(ggml_cgraph* gf, const ggml_tensor* tensor) {
        if (gf == nullptr || tensor == nullptr) {
            return -1;
        }

        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (ggml_graph_node(gf, i) == tensor) {
                return i;
            }
        }
        return -1;
    }

    static bool segment_contains_node(const Segment& segment, int node_index) {
        if (node_index < 0) {
            return false;
        }

        return std::find(segment.internal_node_indices.begin(),
                         segment.internal_node_indices.end(),
                         node_index) != segment.internal_node_indices.end();
    }

    Plan attach_comm_ops_to_plan(ggml_cgraph* gf,
                                 const Plan& base_plan,
                                 const char* log_desc) {
        Plan plan = base_plan;
        for (Segment& segment : plan.segments) {
            segment.comm_ops.clear();
        }

        if (gf == nullptr || g_comm_marks.empty() || plan.segments.empty()) {
            return plan;
        }

        size_t attached_count = 0;
        for (const CommMark& mark : g_comm_marks) {
            const int input_idx = node_index_for_tensor(gf, mark.input);
            const int output_idx = mark.kind == Segment::CommKind::BROADCAST
                                       ? -1
                                       : node_index_for_tensor(gf, mark.output);

            if (input_idx < 0 || (mark.kind != Segment::CommKind::BROADCAST && output_idx < 0)) {
                if (log_desc != nullptr) {
                    LOG_WARN("%s graph cut comm mark skipped: name=%s kind=%s input_idx=%d output_idx=%d",
                             log_desc,
                             mark.name.c_str(),
                             comm_kind_name(mark.kind),
                             input_idx,
                             output_idx);
                }
                continue;
            }

            Segment* output_segment = nullptr;
            if (output_idx >= 0) {
                for (Segment& segment : plan.segments) {
                    if (std::find(segment.output_node_indices.begin(),
                                  segment.output_node_indices.end(),
                                  output_idx) != segment.output_node_indices.end()) {
                        output_segment = &segment;
                        break;
                    }
                }
            }

            Segment* target_segment = output_segment;
            if (target_segment != nullptr) {
                if (std::find(target_segment->seed_node_indices.begin(),
                              target_segment->seed_node_indices.end(),
                              input_idx) == target_segment->seed_node_indices.end()) {
                    target_segment->seed_node_indices.push_back(input_idx);
                }
            }

            for (Segment& segment : plan.segments) {
                if (target_segment == nullptr && segment_contains_node(segment, input_idx)) {
                    target_segment = &segment;
                    break;
                }
            }

            if (target_segment == nullptr) {
                if (log_desc != nullptr) {
                    LOG_WARN("%s graph cut comm mark skipped: name=%s kind=%s input node %d is not internal to any segment",
                             log_desc,
                             mark.name.c_str(),
                             comm_kind_name(mark.kind),
                             input_idx);
                }
                continue;
            }

            Segment::CommOp op;
            op.kind = mark.kind;
            op.name = mark.name;
            op.input_node_index = input_idx;
            op.output_node_index = output_idx;
            op.count_per_peer = mark.count_per_peer;
            op.reduce_op = mark.reduce_op;
            op.root = mark.root;
            target_segment->comm_ops.push_back(std::move(op));
            ++attached_count;
        }

        if (attached_count > 0 && log_desc != nullptr) {
            LOG_DEBUG("%s graph cut attached %zu comm op(s) from %zu mark(s)",
                      log_desc,
                      attached_count,
                      g_comm_marks.size());
        }

        return plan;
    }

    int leaf_count(ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);
        return gf->n_leafs;
    }

    ggml_tensor* leaf_tensor(ggml_cgraph* gf, int leaf_index) {
        GGML_ASSERT(gf != nullptr);
        if (leaf_index < 0 || leaf_index >= gf->n_leafs) {
            return nullptr;
        }
        return gf->leafs[leaf_index];
    }

    ggml_backend_buffer_t tensor_buffer(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    }

    ggml_tensor* cache_source_tensor(ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return tensor->view_src ? tensor->view_src : tensor;
    }

    size_t cache_tensor_bytes(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return 0;
        }
        const ggml_tensor* cache_src = tensor->view_src ? tensor->view_src : tensor;
        return ggml_nbytes(cache_src);
    }

    bool plan_matches_graph(ggml_cgraph* gf, const Plan& plan) {
        GGML_ASSERT(gf != nullptr);
        if (ggml_graph_n_nodes(gf) != plan.n_nodes || gf->n_leafs != plan.n_leafs) {
            return false;
        }
        for (const auto& input_shape_ref : plan.input_shapes) {
            if (input_shape_ref.leaf_index < 0 || input_shape_ref.leaf_index >= gf->n_leafs) {
                return false;
            }
            ggml_tensor* leaf = gf->leafs[input_shape_ref.leaf_index];
            if (leaf == nullptr || input_shape_ref.type != leaf->type) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (input_shape_ref.ne[static_cast<size_t>(d)] != leaf->ne[d]) {
                    return false;
                }
            }
        }
        return true;
    }

    ggml_tensor* output_tensor(ggml_cgraph* gf, const Segment& segment, size_t output_index) {
        GGML_ASSERT(gf != nullptr);
        if (output_index >= segment.output_node_indices.size()) {
            return nullptr;
        }
        int node_index = segment.output_node_indices[output_index];
        if (node_index < 0 || node_index >= ggml_graph_n_nodes(gf)) {
            return nullptr;
        }
        return ggml_graph_node(gf, node_index);
    }

    ggml_tensor* input_tensor(ggml_cgraph* gf, const Segment::InputRef& input_ref) {
        GGML_ASSERT(gf != nullptr);
        if (input_ref.type == Segment::INPUT_PREVIOUS_CUT) {
            if (input_ref.node_index < 0 || input_ref.node_index >= ggml_graph_n_nodes(gf)) {
                return nullptr;
            }
            return ggml_graph_node(gf, input_ref.node_index);
        }
        if (input_ref.leaf_index < 0 || input_ref.leaf_index >= gf->n_leafs) {
            return nullptr;
        }
        return leaf_tensor(gf, input_ref.leaf_index);
    }

    std::vector<ggml_tensor*> param_tensors(ggml_cgraph* gf, const Segment& segment) {
        GGML_ASSERT(gf != nullptr);
        std::vector<ggml_tensor*> tensors;
        std::unordered_set<ggml_tensor*> seen_tensors;
        tensors.reserve(segment.input_refs.size());
        seen_tensors.reserve(segment.input_refs.size());
        for (const auto& input_ref : segment.input_refs) {
            if (input_ref.type != Segment::INPUT_PARAM) {
                continue;
            }
            ggml_tensor* tensor = input_tensor(gf, input_ref);
            if (tensor == nullptr) {
                continue;
            }
            if (seen_tensors.insert(tensor).second) {
                tensors.push_back(tensor);
            }
        }
        return tensors;
    }

    std::vector<ggml_tensor*> runtime_param_tensors(ggml_cgraph* gf, const Segment& segment, const char* log_desc) {
        std::vector<ggml_tensor*> tensors = param_tensors(gf, segment);
        std::vector<ggml_tensor*> filtered_tensors;
        filtered_tensors.reserve(tensors.size());
        for (ggml_tensor* tensor : tensors) {
            if (tensor_buffer(tensor) == nullptr) {
                LOG_WARN("%s graph cut skipping param input without buffer: segment=%s tensor=%s",
                         log_desc == nullptr ? "unknown" : log_desc,
                         segment.group_name.c_str(),
                         tensor->name);
                continue;
            }
            filtered_tensors.push_back(tensor);
        }
        return filtered_tensors;
    }

    std::unordered_set<std::string> collect_future_input_names(ggml_cgraph* gf,
                                                               const Plan& plan,
                                                               size_t current_segment_index) {
        GGML_ASSERT(gf != nullptr);
        std::unordered_set<std::string> future_input_names;
        for (size_t seg_idx = current_segment_index + 1; seg_idx < plan.segments.size(); ++seg_idx) {
            const auto& segment = plan.segments[seg_idx];
            for (const auto& input_ref : segment.input_refs) {
                if (input_ref.type != Segment::INPUT_PREVIOUS_CUT) {
                    continue;
                }
                ggml_tensor* current_input = input_tensor(gf, input_ref);
                if (current_input != nullptr && current_input->name[0] != '\0') {
                    future_input_names.insert(current_input->name);
                }
            }
        }
        return future_input_names;
    }

    ggml_cgraph* build_segment_graph(ggml_cgraph* gf,
                                     const Segment& segment,
                                     ggml_context** graph_ctx_out) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(graph_ctx_out != nullptr);

        // Segment nodes can still reference auxiliary tensors that are neither
        // segment outputs nor explicit input refs (for example receive
        // placeholder source/dependency tensors used by SP communication). Keep the
        // segment graph hash table large enough for every tensor reachable from
        // the original graph instead of sizing it only from the compact segment
        // metadata.
        const size_t graph_size = std::max(segment.internal_node_indices.size() + segment.input_refs.size() + 8,
                                           static_cast<size_t>(ggml_graph_n_nodes(gf) + gf->n_leafs + 16));
        ggml_init_params params = {
            /*.mem_size   =*/ggml_graph_overhead_custom(graph_size, false) + 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context* graph_ctx = ggml_init(params);
        GGML_ASSERT(graph_ctx != nullptr);
        ggml_cgraph* segment_graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
        GGML_ASSERT(segment_graph != nullptr);

        std::unordered_set<ggml_tensor*> segment_internal_tensors;
        segment_internal_tensors.reserve(segment.internal_node_indices.size());
        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node != nullptr) {
                segment_internal_tensors.insert(node);
            }
        }

        std::unordered_set<ggml_tensor*> segment_leaf_tensors;
        std::function<void(ggml_tensor*)> add_segment_leaf = [&](ggml_tensor* tensor) {
            if (tensor == nullptr || segment_internal_tensors.find(tensor) != segment_internal_tensors.end()) {
                return;
            }
            const bool inserted = segment_leaf_tensors.insert(tensor).second;
            if (!inserted) {
                return;
            }
            GGML_ASSERT(segment_graph->n_leafs < segment_graph->size);
            segment_graph->leafs[segment_graph->n_leafs++] = tensor;
            add_segment_leaf(tensor->view_src);
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                add_segment_leaf(tensor->src[src_idx]);
            }
        };

        for (const auto& input : segment.input_refs) {
            add_segment_leaf(input_tensor(gf, input));
        }
        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr) {
                continue;
            }
            if (node->view_src != nullptr) {
                add_segment_leaf(node->view_src);
            }
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ggml_tensor* src = node->src[src_idx];
                add_segment_leaf(src);
                if (src != nullptr && src->view_src != nullptr) {
                    add_segment_leaf(src->view_src);
                }
            }
        }

        for (int output_node_index : segment.output_node_indices) {
            ggml_tensor* output = ggml_graph_node(gf, output_node_index);
            if (output == nullptr) {
                continue;
            }
            ggml_set_output(output);
        }
        for (const auto& comm_op : segment.comm_ops) {
            ggml_tensor* input = comm_input_tensor(gf, comm_op);
            if (input != nullptr) {
                // Communication runs from the post-compute callback. Keep the
                // send tensor alive after graph compute instead of letting the
                // allocator recycle its buffer as an internal temporary.
                ggml_set_output(input);
            }
            ggml_tensor* output = comm_output_tensor(gf, comm_op);
            if (output != nullptr) {
                ggml_set_output(output);
            }
        }
        for (int node_idx : segment.internal_node_indices) {
            ggml_graph_add_node(segment_graph, ggml_graph_node(gf, node_idx));
        }
        rebuild_segment_graph_usage(segment_graph);
        *graph_ctx_out = graph_ctx;
        return segment_graph;
    }

    size_t measure_segment_compute_buffer(ggml_backend_t backend,
                                          ggml_cgraph* gf,
                                          const Segment& segment,
                                          const char* log_desc) {
        GGML_ASSERT(backend != nullptr);
        GGML_ASSERT(gf != nullptr);
        if (segment.internal_node_indices.empty()) {
            return 0;
        }

        ggml_context* graph_ctx    = nullptr;
        ggml_cgraph* segment_graph = build_segment_graph(gf, segment, &graph_ctx);
        ggml_gallocr_t allocr      = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        size_t sizes[1] = {0};
        ggml_gallocr_reserve_n_size(
            allocr,
            segment_graph,
            nullptr,
            nullptr,
            sizes);
        size_t buffer_size = sizes[0];

        ggml_gallocr_free(allocr);
        ggml_free(graph_ctx);
        return buffer_size;
    }

    Plan build_plan(ggml_backend_t backend,
                    ggml_cgraph* gf,
                    const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                    const char* log_desc) {
        GGML_ASSERT(backend != nullptr);
        GGML_ASSERT(gf != nullptr);
        Plan plan;
        plan.available    = true;
        const int n_nodes = ggml_graph_n_nodes(gf);
        if (n_nodes <= 0) {
            return plan;
        }
        plan.n_nodes = n_nodes;
        plan.n_leafs = gf->n_leafs;
        for (int i = 0; i < gf->n_leafs; ++i) {
            ggml_tensor* leaf = gf->leafs[i];
            if (is_params_tensor(params_tensor_set, leaf)) {
                continue;
            }
            auto shape       = input_shape(leaf);
            shape.leaf_index = i;
            plan.input_shapes.push_back(shape);
        }

        std::unordered_map<const ggml_tensor*, int> producer_index;
        producer_index.reserve(static_cast<size_t>(n_nodes));
        for (int i = 0; i < n_nodes; ++i) {
            producer_index[ggml_graph_node(gf, i)] = i;
        }

        std::vector<Segment> grouped_segments;
        std::unordered_map<std::string, size_t> group_to_segment;
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (!is_graph_cut_tensor(node)) {
                continue;
            }

            plan.has_cuts = true;
            std::string full_name(node->name);
            std::string payload = full_name.substr(std::strlen(GGML_RUNNER_CUT_PREFIX));
            size_t sep          = payload.find('|');
            std::string group   = sep == std::string::npos ? payload : payload.substr(0, sep);

            auto it = group_to_segment.find(group);
            if (it == group_to_segment.end()) {
                Segment segment;
                segment.group_name = group;
                segment.output_node_indices.push_back(i);
                group_to_segment[group] = grouped_segments.size();
                grouped_segments.push_back(std::move(segment));
            } else {
                auto& segment = grouped_segments[it->second];
                segment.output_node_indices.push_back(i);
            }
        }

        if (!plan.has_cuts) {
            return plan;
        }

        for (const CommMark& mark : g_comm_marks) {
            if (mark.kind == Segment::CommKind::BROADCAST || mark.input == nullptr || mark.output == nullptr) {
                continue;
            }
            const int input_idx = node_index_for_tensor(gf, mark.input);
            const int output_idx = node_index_for_tensor(gf, mark.output);
            if (input_idx < 0 || output_idx < 0) {
                continue;
            }
            for (Segment& segment : grouped_segments) {
                if (std::find(segment.output_node_indices.begin(),
                              segment.output_node_indices.end(),
                              output_idx) == segment.output_node_indices.end()) {
                    continue;
                }
                if (std::find(segment.seed_node_indices.begin(),
                              segment.seed_node_indices.end(),
                              input_idx) == segment.seed_node_indices.end()) {
                    segment.seed_node_indices.push_back(input_idx);
                }
                break;
            }
        }

        grouped_segments = topo_sort_segments(gf,
                                              grouped_segments,
                                              producer_index,
                                              log_desc);

        std::unordered_set<int> available_cut_output_node_indices;
        available_cut_output_node_indices.reserve(static_cast<size_t>(n_nodes));
        for (auto& segment : grouped_segments) {
            build_segment(gf,
                          plan,
                          segment,
                          producer_index,
                          available_cut_output_node_indices,
                          backend,
                          params_tensor_set,
                          log_desc);
        }

        ggml_tensor* final_output = ggml_graph_node(gf, -1);
        if (final_output != nullptr && available_cut_output_node_indices.find(n_nodes - 1) == available_cut_output_node_indices.end()) {
            Segment final_segment;
            final_segment.group_name = "ggml_runner.final";
            final_segment.output_node_indices.push_back(n_nodes - 1);
            build_segment(gf,
                          plan,
                          final_segment,
                          producer_index,
                          available_cut_output_node_indices,
                          backend,
                          params_tensor_set,
                          log_desc);
        }

        populate_future_input_names(plan);
        return plan;
    }

    Plan apply_max_vram_budget(ggml_cgraph* gf,
                               const Plan& base_plan,
                               size_t max_graph_vram_bytes,
                               ggml_backend_t backend,
                               const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                               const char* log_desc) {
        GGML_ASSERT(backend != nullptr);
        GGML_ASSERT(gf != nullptr);
        int64_t t_budget_begin = ggml_time_ms();
        if (max_graph_vram_bytes == 0 || !base_plan.has_cuts || base_plan.segments.size() <= 1) {
            return base_plan;
        }

        const int n_nodes = ggml_graph_n_nodes(gf);
        std::unordered_map<const ggml_tensor*, int> producer_index;
        producer_index.reserve(static_cast<size_t>(n_nodes));
        for (int i = 0; i < n_nodes; ++i) {
            producer_index[ggml_graph_node(gf, i)] = i;
        }

        Plan merged_plan;
        merged_plan.available = true;
        merged_plan.has_cuts  = base_plan.has_cuts;
        merged_plan.valid     = base_plan.valid;
        merged_plan.n_nodes   = base_plan.n_nodes;
        merged_plan.n_leafs   = base_plan.n_leafs;

        std::unordered_set<int> available_cut_output_node_indices;
        available_cut_output_node_indices.reserve(static_cast<size_t>(n_nodes));

        size_t start_segment_index = 0;
        while (start_segment_index < base_plan.segments.size()) {
            Plan single_plan;
            auto single_available_cut_output_node_indices = available_cut_output_node_indices;
            auto single_seed                              = make_segment_seed(base_plan,
                                                                              start_segment_index,
                                                                              start_segment_index);
            build_segment(gf,
                          single_plan,
                          single_seed,
                          producer_index,
                          single_available_cut_output_node_indices,
                          backend,
                          params_tensor_set,
                          log_desc);
            GGML_ASSERT(!single_plan.segments.empty());

            size_t best_end_segment_index = start_segment_index;
            bool can_merge_next_segment   = graph_cut_segment_vram_bytes(single_plan.segments.back()) <= max_graph_vram_bytes;

            while (can_merge_next_segment && best_end_segment_index + 1 < base_plan.segments.size()) {
                const size_t next_end_segment_index = best_end_segment_index + 1;
                Plan candidate_plan;
                auto candidate_available_cut_output_node_indices = available_cut_output_node_indices;
                auto candidate_seed                              = make_segment_seed(base_plan,
                                                                                     start_segment_index,
                                                                                     next_end_segment_index);
                build_segment(gf,
                              candidate_plan,
                              candidate_seed,
                              producer_index,
                              candidate_available_cut_output_node_indices,
                              backend,
                              params_tensor_set,
                              log_desc);
                GGML_ASSERT(!candidate_plan.segments.empty());

                const auto& candidate_segment = candidate_plan.segments.back();
                if (graph_cut_segment_vram_bytes(candidate_segment) > max_graph_vram_bytes) {
                    break;
                }

                best_end_segment_index = next_end_segment_index;
            }

            auto best_seed = make_segment_seed(base_plan,
                                               start_segment_index,
                                               best_end_segment_index);
            build_segment(gf,
                          merged_plan,
                          best_seed,
                          producer_index,
                          available_cut_output_node_indices,
                          backend,
                          params_tensor_set,
                          log_desc);
            start_segment_index = best_end_segment_index + 1;
        }

        if (log_desc != nullptr && merged_plan.segments.size() != base_plan.segments.size()) {
            LOG_INFO("%s graph cut max_vram=%.2f MB merged %zu segments -> %zu segments",
                     log_desc,
                     max_graph_vram_bytes / 1024.0 / 1024.0,
                     base_plan.segments.size(),
                     merged_plan.segments.size());
        }

        if (log_desc != nullptr) {
            LOG_INFO("%s graph cut max_vram budget merge took %lld ms",
                     log_desc,
                     ggml_time_ms() - t_budget_begin);
        }

        populate_future_input_names(merged_plan);
        return merged_plan;
    }

    Plan resolve_plan(ggml_backend_t backend,
                      ggml_cgraph* gf,
                      PlanCache* cache,
                      size_t max_graph_vram_bytes,
                      const std::unordered_set<const ggml_tensor*>& params_tensor_set,
                      const char* log_desc) {
        GGML_ASSERT(backend != nullptr);
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(cache != nullptr);

        int64_t t_prepare_begin = ggml_time_ms();
        Plan base_plan;
        int64_t t_plan_begin = ggml_time_ms();
        if (cache->graph_cut_plan.available && plan_matches_graph(gf, cache->graph_cut_plan)) {
            base_plan = cache->graph_cut_plan;
        } else {
            base_plan                                = build_plan(backend, gf, params_tensor_set, log_desc);
            cache->graph_cut_plan                    = base_plan;
            cache->graph_cut_plan.available          = true;
            cache->budgeted_graph_cut_plan.available = false;
        }

        Plan resolved_plan = base_plan;
        if (max_graph_vram_bytes > 0 && base_plan.has_cuts) {
            if (cache->budgeted_graph_cut_plan.available &&
                cache->budgeted_graph_cut_plan_max_vram_bytes == max_graph_vram_bytes &&
                plan_matches_graph(gf, cache->budgeted_graph_cut_plan)) {
                resolved_plan = cache->budgeted_graph_cut_plan;
            } else {
                resolved_plan                                 = apply_max_vram_budget(gf,
                                                                                      base_plan,
                                                                                      max_graph_vram_bytes,
                                                                                      backend,
                                                                                      params_tensor_set,
                                                                                      log_desc);
                cache->budgeted_graph_cut_plan                = resolved_plan;
                cache->budgeted_graph_cut_plan.available      = true;
                cache->budgeted_graph_cut_plan_max_vram_bytes = max_graph_vram_bytes;
            }
        }
        return attach_comm_ops_to_plan(gf, resolved_plan, log_desc);
    }

    const char* comm_kind_name(Segment::CommKind kind) {
        switch (kind) {
            case Segment::CommKind::NONE:
                return "none";
            case Segment::CommKind::ALL_REDUCE:
                return "all_reduce";
            case Segment::CommKind::ALL_GATHER:
                return "all_gather";
            case Segment::CommKind::ALL_TO_ALL:
                return "all_to_all";
            case Segment::CommKind::BROADCAST:
                return "broadcast";
        }
        return "unknown";
    }

    ggml_tensor* comm_input_tensor(ggml_cgraph* gf,
                                   const Segment::CommOp& comm_op) {
        if (gf == nullptr) {
            return nullptr;
        }
        if (comm_op.input_node_index < 0 || comm_op.input_node_index >= gf->n_nodes) {
            return nullptr;
        }
        return gf->nodes[comm_op.input_node_index];
    }

    ggml_tensor* comm_output_tensor(ggml_cgraph* gf,
                                    const Segment::CommOp& comm_op) {
        if (gf == nullptr) {
            return nullptr;
        }
        if (comm_op.output_node_index < 0 || comm_op.output_node_index >= gf->n_nodes) {
            return nullptr;
        }
        return gf->nodes[comm_op.output_node_index];
    }

    static void set_comm_error(std::string* error, const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
    }

    static std::string comm_op_display_name(const Segment::CommOp& comm_op) {
        if (!comm_op.name.empty()) {
            return comm_op.name;
        }
        return comm_kind_name(comm_op.kind);
    }

    static bool check_comm_input(ggml_cgraph* gf,
                                 const Segment::CommOp& comm_op,
                                 ggml_tensor** input_out,
                                 std::string* error) {
        ggml_tensor* input = comm_input_tensor(gf, comm_op);
        if (input == nullptr) {
            set_comm_error(error,
                           sd_format("graph cut comm op %s has invalid input_node_index=%d",
                                     comm_op_display_name(comm_op).c_str(),
                                     comm_op.input_node_index));
            return false;
        }

        if (input_out != nullptr) {
            *input_out = input;
        }
        return true;
    }

    static bool check_comm_output(ggml_cgraph* gf,
                                  const Segment::CommOp& comm_op,
                                  ggml_tensor** output_out,
                                  std::string* error) {
        ggml_tensor* output = comm_output_tensor(gf, comm_op);
        if (output == nullptr) {
            set_comm_error(error,
                           sd_format("graph cut comm op %s has invalid output_node_index=%d",
                                     comm_op_display_name(comm_op).c_str(),
                                     comm_op.output_node_index));
            return false;
        }

        if (output_out != nullptr) {
            *output_out = output;
        }
        return true;
    }

    bool execute_comm_op(edgedit::parallel::ProcessGroup& group,
                         ggml_cgraph* gf,
                         const Segment::CommOp& comm_op,
                         std::string* error) {
        if (!comm_op.enabled) {
            return true;
        }

        if (gf == nullptr) {
            set_comm_error(error, "execute_comm_op failed: graph is null");
            return false;
        }

        if (!group.enabled()) {
            // world_size == 1 或 parallel disabled 时，通信可以视为 no-op。
            return true;
        }

        if (comm_op.kind == Segment::CommKind::NONE) {
            return true;
        }

        ggml_tensor* input = nullptr;
        ggml_tensor* output = nullptr;

        if (!check_comm_input(gf, comm_op, &input, error)) {
            return false;
        }

        const char* kind_name = comm_kind_name(comm_op.kind);
        const std::string op_name = comm_op_display_name(comm_op);

        switch (comm_op.kind) {
            case Segment::CommKind::NONE:
                return true;

            case Segment::CommKind::ALL_REDUCE: {
                if (!check_comm_output(gf, comm_op, &output, error)) {
                    return false;
                }

                std::string comm_error;
                bool ok = edgedit::ggml_comm::all_reduce(group,
                                                          input,
                                                          output,
                                                          comm_op.reduce_op,
                                                          &comm_error);
                if (!ok) {
                    set_comm_error(error,
                                   sd_format("graph cut comm op %s(%s) failed: %s",
                                             op_name.c_str(),
                                             kind_name,
                                             comm_error.c_str()));
                    return false;
                }
                return true;
            }

            case Segment::CommKind::ALL_GATHER: {
                if (!check_comm_output(gf, comm_op, &output, error)) {
                    return false;
                }

                std::string comm_error;
                bool ok = edgedit::ggml_comm::all_gather(group,
                                                          input,
                                                          output,
                                                          &comm_error);
                if (!ok) {
                    set_comm_error(error,
                                   sd_format("graph cut comm op %s(%s) failed: %s",
                                             op_name.c_str(),
                                             kind_name,
                                             comm_error.c_str()));
                    return false;
                }
                return true;
            }

            case Segment::CommKind::ALL_TO_ALL: {
                if (!check_comm_output(gf, comm_op, &output, error)) {
                    return false;
                }

                if (comm_op.count_per_peer == 0) {
                    set_comm_error(error,
                                   sd_format("graph cut comm op %s(%s) failed: count_per_peer is 0",
                                             op_name.c_str(),
                                             kind_name));
                    return false;
                }

                std::string comm_error;
                bool ok = edgedit::ggml_comm::all_to_all(group,
                                                          input,
                                                          output,
                                                          comm_op.count_per_peer,
                                                          &comm_error);
                if (!ok) {
                    set_comm_error(error,
                                   sd_format("graph cut comm op %s(%s) failed: %s",
                                             op_name.c_str(),
                                             kind_name,
                                             comm_error.c_str()));
                    return false;
                }
                return true;
            }

            case Segment::CommKind::BROADCAST: {
                std::string comm_error;
                bool ok = edgedit::ggml_comm::broadcast(group,
                                                         input,
                                                         comm_op.root,
                                                         &comm_error);
                if (!ok) {
                    set_comm_error(error,
                                   sd_format("graph cut comm op %s(%s) failed: %s",
                                             op_name.c_str(),
                                             kind_name,
                                             comm_error.c_str()));
                    return false;
                }
                return true;
            }
        }

        set_comm_error(error,
                       sd_format("graph cut comm op %s has unsupported kind=%d",
                                 op_name.c_str(),
                                 static_cast<int>(comm_op.kind)));
        return false;
    }

    bool execute_segment_comm_ops(edgedit::parallel::ProcessGroup& group,
                                  ggml_cgraph* gf,
                                  const Segment& segment,
                                  std::string* error) {
        if (segment.comm_ops.empty()) {
            return true;
        }

        for (size_t i = 0; i < segment.comm_ops.size(); ++i) {
            const Segment::CommOp& comm_op = segment.comm_ops[i];

            std::string comm_error;
            if (!execute_comm_op(group, gf, comm_op, &comm_error)) {
                set_comm_error(error,
                               sd_format("execute segment comm ops failed at index=%zu, group=%s, error=%s",
                                         i,
                                         segment.group_name.c_str(),
                                         comm_error.c_str()));
                return false;
            }
        }

        return true;
    }
    
}  // namespace sd::ggml_graph_cut
