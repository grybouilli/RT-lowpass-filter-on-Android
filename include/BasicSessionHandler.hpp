#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

class BasicSessionHandler {
public:

    BasicSessionHandler(const std::string model_filename)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "lowpass_rnn\n")
    {
        m_session_options.SetIntraOpNumThreads(1);
        m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        m_session = Ort::Session(m_env, model_filename.c_str(), m_session_options);
    }

    Ort::Session& session() { return m_session.value(); }
    
private:
    Ort::Env m_env;
    Ort::SessionOptions m_session_options;
    std::optional<Ort::Session> m_session;
};