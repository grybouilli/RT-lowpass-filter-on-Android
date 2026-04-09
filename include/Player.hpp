#pragma once

#include <oboe/Oboe.h>
#include <AudioParams.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <GRUInference.hpp>
#include <BasicSessionHandler.hpp>
#include <npy.hpp>
#include <chrono>
// ---------------------------------------------------------------------------
// Player
//   Oboe output callback.  Reads raw audio from SharedAudioBuffer, runs it
//   through the GRU filter via GRUBinding (zero hot-path allocations), and
//   writes filtered samples back to Oboe.
// ---------------------------------------------------------------------------
template<IsIRRGRUInfo IRRGRU>
class Player : public oboe::AudioStreamDataCallback {
public:
    Player(
        const IRRGRU&  gru,
        const float    fc_normed,
        int32_t        sample_rate,
        int32_t        channels,
        audio_buffer&  buffer,
        const bool     debug = false,
        const bool     profiling = false
    )
        : m_sample_rate    { sample_rate }
        , m_channels       { channels }
        , m_buffer         { buffer }
        , m_session_handle { "./lowpass_rnn.onnx" }
        // GRUBinding is constructed after the session so we can pass the session ref
        , m_gru_binding    { m_session_handle.session(), gru, fc_normed }
        , m_expected_frames { static_cast<int32_t>(gru.buffer_size()) }
        , m_debug { debug }
        , m_profiling { profiling }
    {
        printf("normed frequency is %f\n", fc_normed);
    }

    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream* /*stream*/,
        void*    audio_data,
        int32_t  num_frames) override
    {
        using namespace std::chrono;
        audio_sample_t* out = static_cast<audio_sample_t*>(audio_data);
        const int32_t   num_samples = num_frames * m_channels;

        // Read raw input from the shared ring-buffer
        m_buffer.read(out, num_samples);

        // Guard: GRU binding is sized for a fixed buffer_size; skip inference
        // if Oboe gives us an unexpected frame count to avoid UB.
        if (num_frames != m_expected_frames) {
            fprintf(stderr,
                "Player: unexpected frame count %d (expected %d), passing through\n",
                num_frames, m_expected_frames);
            return oboe::DataCallbackResult::Continue;
        }

        bool ret;
        decltype(std::chrono::high_resolution_clock::now()) start, end;
        if(m_profiling)
        {
            start = std::chrono::high_resolution_clock::now();
        }

        // Run inference — writes filtered samples back into `out` in-place
        ret = m_gru_binding.run(out, out);

        if(m_profiling)
        {
            end = std::chrono::high_resolution_clock::now();
            m_recorded_performances.push_back(duration_cast<milliseconds>(end-start).count());
        }
        if(ret && m_debug){
            // Record output for offline dump
            m_recorded_signal.insert(
                m_recorded_signal.end(),
                m_gru_binding.output_ptr(),
                m_gru_binding.output_ptr() + num_samples
            );
        }

        return oboe::DataCallbackResult::Continue;
    }

    void dump_debug(const std::string& filename)
    {
        npy::npy_data<audio_sample_t> d;
        d.data  = m_recorded_signal;
        d.shape = { m_recorded_signal.size() };
        npy::write_npy(filename, d);
    }

    void dump_profiling(const std::string& filename)
    {
        npy::npy_data<double> d;
        d.data  = m_recorded_performances;
        d.shape = { m_recorded_performances.size() };
        npy::write_npy(filename, d);
    }

private:
    int32_t              m_sample_rate;
    int32_t              m_channels;
    audio_buffer&        m_buffer;

    BasicSessionHandler  m_session_handle;
    GRUBinding<IRRGRU>   m_gru_binding;

    int32_t              m_expected_frames;

    bool                m_debug;
    std::vector<audio_sample_t> m_recorded_signal;
    
    bool                m_profiling;
    std::vector<double>  m_recorded_performances;
};