#pragma once

#include <string>
enum class SupportedInferenceEngines { Ort, Anira_with_Ort };

struct GeneralInferenceParams {
    std::string               model_filename;
    bool                      debug_mode_on;
    SupportedInferenceEngines chosen_engine;
    float                     Fc_normed;
};