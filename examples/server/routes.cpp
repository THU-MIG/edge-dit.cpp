#include "routes.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

void set_json_response(httplib::Response& res, const json& body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(2), "application/json; charset=utf-8");
}

void set_error_response(httplib::Response& res, int status, const std::string& message) {
    json body;
    body["error"] = {
        {"message", message},
        {"status", status},
    };
    set_json_response(res, body, status);
}

json parse_json_body(const httplib::Request& req, std::string* error) {
    if (req.body.empty()) {
        if (error != nullptr) {
            *error = "empty request body";
        }
        return json();
    }
    try {
        return json::parse(req.body);
    } catch (const std::exception& e) {
        if (error != nullptr) {
            *error = std::string("invalid JSON: ") + e.what();
        }
        return json();
    }
}

json image_metadata_json(const ed_image_t& image, int index) {
    return {
        {"index", index},
        {"width", image.width},
        {"height", image.height},
        {"channels", image.channels},
        {"format", "png"},
    };
}

template <typename Handler>
void register_get_aliases(httplib::Server& server, const char* path, Handler&& handler) {
    server.Get(path, handler);
    server.Get(std::string("/edgedit") + std::string(path + 3), handler);
    server.Get(std::string("/edge-dit") + std::string(path + 3), handler);
}

template <typename Handler>
void register_post_aliases(httplib::Server& server, const char* path, Handler&& handler) {
    server.Post(path, handler);
    server.Post(std::string("/edgedit") + std::string(path + 3), handler);
    server.Post(std::string("/edge-dit") + std::string(path + 3), handler);
}

}  // namespace

void register_edgedit_routes(httplib::Server& server, EdgeDitServerRuntime& runtime) {
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        json body;
        body["service"] = "edge-dit";
        body["message"] = "edge-dit server is running";
        body["health"] = "/ed/v1/health";
        body["capabilities"] = "/ed/v1/capabilities";
        body["image_generation"] = "/ed/v1/images/generations";
        body["video_generation"] = "/ed/v1/videos/generations";
        body["aliases"] = {"/edgedit/v1", "/edge-dit/v1"};
        set_json_response(res, body);
    });

    register_get_aliases(server, "/ed/v1/health", [&runtime](const httplib::Request&, httplib::Response& res) {
        json body;
        body["status"] = runtime.ctx != nullptr ? "ok" : "error";
        body["service"] = "edge-dit";
        body["model"] = runtime.display_model_path;
        set_json_response(res, body, runtime.ctx != nullptr ? 200 : 503);
    });

    register_get_aliases(server, "/ed/v1/models", [&runtime](const httplib::Request&, httplib::Response& res) {
        json body;
        body["object"] = "list";
        body["data"] = json::array({
            {
                {"id", runtime.display_model_path.empty() ? "edge-dit-model" : runtime.display_model_path},
                {"object", "model"},
                {"backend", "edge-dit"},
            },
        });
        set_json_response(res, body);
    });

    register_get_aliases(server, "/ed/v1/capabilities", [&runtime](const httplib::Request&, httplib::Response& res) {
        set_json_response(res, build_capabilities_response(runtime));
    });

    register_post_aliases(server, "/ed/v1/images/generations", [&runtime](const httplib::Request& req, httplib::Response& res) {
        std::string error;
        json body = parse_json_body(req, &error);
        if (!error.empty()) {
            set_error_response(res, 400, error);
            return;
        }
        if (!body.is_object()) {
            set_error_response(res, 400, "request body must be a JSON object");
            return;
        }

        EdgeDitImageRequest image_request;
        if (!build_image_request(body, runtime, &image_request, &error)) {
            set_error_response(res, 400, error);
            return;
        }

        ed_image_batch_t batch = {};
        const auto start = std::chrono::steady_clock::now();
        ed_status_t status = ED_STATUS_ERROR;
        std::string last_error;
        {
            std::lock_guard<std::mutex> lock(*runtime.ctx_mutex);
            status = ed_generate_image(runtime.ctx, &image_request.params, &batch);
            const char* err = ed_get_last_error(runtime.ctx);
            if (err != nullptr) {
                last_error = err;
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

        if (status != ED_STATUS_OK) {
            ed_free_image_batch(&batch);
            if (last_error.empty()) {
                last_error = "image generation failed";
            }
            json err_body;
            err_body["error"] = {
                {"message", last_error},
                {"status", ed_status_to_string(status)},
                {"code", static_cast<int>(status)},
            };
            set_json_response(res, err_body, 500);
            return;
        }

        json images = json::array();
        for (int i = 0; i < batch.count; ++i) {
            std::vector<uint8_t> png;
            if (!image_to_png_bytes(batch.images[i], &png)) {
                ed_free_image_batch(&batch);
                set_error_response(res, 500, "failed to encode generated image as PNG");
                return;
            }
            images.push_back({
                {"b64_png", base64_encode(png)},
                {"metadata", image_metadata_json(batch.images[i], i)},
            });
        }

        json result;
        result["object"] = "edgedit.image_generation";
        result["model"] = runtime.display_model_path;
        result["created_ms"] = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
        result["elapsed_ms"] = elapsed_ms;
        result["parameters"] = {
            {"prompt", image_request.prompt},
            {"width", image_request.params.width},
            {"height", image_request.params.height},
            {"steps", image_request.params.sample.steps},
            {"seed", image_request.params.seed},
            {"cfg_scale", image_request.params.sample.cfg_scale},
            {"image_cfg_scale", image_request.params.sample.image_cfg_scale},
            {"distilled_guidance", image_request.params.sample.distilled_guidance},
            {"flow_shift", image_request.params.sample.flow_shift},
            {"cache_mode", ed_cache_mode_to_string(image_request.params.sample.cache_mode)},
        };
        result["data"] = images;

        ed_free_image_batch(&batch);
        set_json_response(res, result);
    });

    register_post_aliases(server, "/ed/v1/videos/generations", [&runtime](const httplib::Request& req, httplib::Response& res) {
        std::string error; json body=parse_json_body(req,&error);
        if (!error.empty() || !body.is_object()) { set_error_response(res,400,error.empty()?"request body must be a JSON object":error); return; }
        EdgeDitVideoRequest request; if (!build_video_request(body,runtime,&request,&error)) { set_error_response(res,400,error); return; }
        ed_video_t video={}; ed_status_t status=ED_STATUS_ERROR; std::string last_error; const auto start=std::chrono::steady_clock::now();
        { std::lock_guard<std::mutex> lock(*runtime.ctx_mutex); status=ed_generate_video(runtime.ctx,&request.params,&video); const char* err=ed_get_last_error(runtime.ctx); if(err) last_error=err; }
        const double elapsed_ms=std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(std::chrono::steady_clock::now()-start).count();
        if(status!=ED_STATUS_OK){ ed_free_video(&video); json e; e["error"]={{"message",last_error.empty()?"video generation failed":last_error},{"status",ed_status_to_string(status)},{"code",static_cast<int>(status)}}; set_json_response(res,e,500); return; }
        json frames=json::array(); for(int i=0;i<video.frame_count;++i){ std::vector<uint8_t> png; if(!image_to_png_bytes(video.frames[i],&png)){ed_free_video(&video);set_error_response(res,500,"failed to encode generated frame");return;} frames.push_back({{"b64_png",base64_encode(png)},{"metadata",image_metadata_json(video.frames[i],i)}}); }
        json result={{"object","edgedit.video_generation"},{"model",runtime.display_model_path},{"elapsed_ms",elapsed_ms},{"fps",request.params.fps},{"frames",frames}};
        if(video.audio && video.audio_sample_count>0 && video.audio_channels>0){ const size_t n=static_cast<size_t>(video.audio_sample_count)*video.audio_channels*sizeof(float); std::vector<uint8_t> bytes(n); std::memcpy(bytes.data(),video.audio,n); result["audio"]={{"b64_f32le",base64_encode(bytes)},{"sample_rate",video.audio_sample_rate},{"channels",video.audio_channels},{"sample_count",video.audio_sample_count}}; }
        ed_free_video(&video); set_json_response(res,result);
    });
}
