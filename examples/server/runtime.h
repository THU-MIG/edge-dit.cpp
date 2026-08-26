#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "edge-dit.h"
#include "json.hpp"

using json = nlohmann::json;

struct EdgeDitServerParams {
    std::string host = "127.0.0.1";
    int port = 8080;
    bool verbose = false;
};

struct EdgeDitDefaultGenerationParams {
    int width = 1024;
    int height = 1024;
    int frames = 90;
    int fps = 24;
    int steps = 20;
    int64_t seed = -1;
    float cfg_scale = 1.0f;
    float image_cfg_scale = 1.0f;
    float distilled_guidance = 3.5f;
    float flow_shift = 0.0f;
    ed_sampler_t sampler = ED_SAMPLER_AUTO;
    ed_scheduler_t scheduler = ED_SCHEDULER_AUTO;
    ed_cache_mode_t cache_mode = ED_CACHE_DISABLED;
};

struct EdgeDitServerRuntime {
    ed_context_t* ctx = nullptr;
    std::mutex* ctx_mutex = nullptr;
    const EdgeDitServerParams* server = nullptr;
    const ed_context_params_t* context = nullptr;
    const EdgeDitDefaultGenerationParams* defaults = nullptr;
    std::string display_model_path;
};

struct EdgeDitImageRequest {
    ed_image_generation_params_t params = {};
    std::string prompt;
    std::string negative_prompt;
    std::string cache_scm_mask;
};

struct EdgeDitVideoRequest {
    ed_video_generation_params_t params = {};
    std::string prompt;
    std::string negative_prompt;
    std::string cache_scm_mask;
    std::vector<float> hires_sigmas;
    ed_image_t init_image = {};
    ed_image_t end_image = {};
    std::vector<ed_image_t> ref_images;
    std::vector<std::vector<uint8_t>> image_storage;
    std::vector<std::vector<ed_image_t>> ref_video_frames;
    std::vector<ed_ref_video_t> ref_videos;
    std::vector<std::vector<float>> audio_storage;
    std::vector<ed_audio_t> ref_audios;
};

std::string ed_status_to_string(ed_status_t status);
std::string ed_cache_mode_to_string(ed_cache_mode_t mode);
bool ed_cache_mode_from_string(const std::string& text, ed_cache_mode_t* mode);
bool ed_sampler_from_string(const std::string& text, ed_sampler_t* sampler);
bool ed_scheduler_from_string(const std::string& text, ed_scheduler_t* scheduler);

std::string base64_encode(const std::vector<uint8_t>& bytes);
bool image_to_png_bytes(const ed_image_t& image, std::vector<uint8_t>* bytes);

bool build_image_request(const json& body,
                         const EdgeDitServerRuntime& runtime,
                         EdgeDitImageRequest* request,
                         std::string* error);
bool build_video_request(const json& body,
                         const EdgeDitServerRuntime& runtime,
                         EdgeDitVideoRequest* request,
                         std::string* error);

json build_capabilities_response(const EdgeDitServerRuntime& runtime);
