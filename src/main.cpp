#include <AudioParams.hpp>
#include <Recorder.hpp>
#include <Player.hpp>
#include <acquire_audio_stream.hpp>
#include <signal.h>
#include <IIRGRUInfo.hpp>
#include <IIRGRUUtils.hpp>
#include <onnxruntime_cxx_api.h>

static constexpr int KDurationMs = 10000;
std::atomic<bool> run = true;
static constexpr int32_t audio_buffer_size = 192;

void sigint_handler(int arg)
{
    run = false;
}

int main(int argc, char ** argv)
{
    int32_t fc = 150;
    if(argc > 1)
    {
        fc = atoi(argv[1]);
    }
    signal(SIGINT, sigint_handler);

    audio_buffer buffer(4096);
    oboe::AudioStreamBuilder in_builder, out_builder;
    int32_t in_sr, out_sr;

    // Acquire input and output streams
    if( in_sr = acquire_audio_stream(
            in_builder,
            oboe::Direction::Input,
            oboe::PerformanceMode::LowLatency,
            oboe::SharingMode::Exclusive,
            oboe::AudioFormat::Float,
            oboe::ChannelCount::Mono
        ); in_sr < 0
    ) {
        fprintf(stderr, "Could not acquire input stream\n");
        return  EXIT_FAILURE;
    }

    if( out_sr = acquire_audio_stream(
            out_builder,
            oboe::Direction::Output,
            oboe::PerformanceMode::LowLatency,
            oboe::SharingMode::Exclusive,
            oboe::AudioFormat::Float,
            oboe::ChannelCount::Mono
        ); out_sr < 0
    ) {
        fprintf(stderr, "Could not acquire output stream\n");
        return  EXIT_FAILURE;
    }

    // GRU related stuff
    IRRGRUInfo<1, audio_buffer_size, 2, 64, 2> gru;

    // Instanciate recorder and player, and attach them to respectful streams
    Recorder recorder (in_sr, 1, buffer);
    Player player (gru, normalize_frequency((float)fc, (float)out_sr), out_sr, 1, buffer);

    in_builder.setDataCallback(&recorder)->setSampleRate(in_sr)->setFramesPerCallback(audio_buffer_size);;
    out_builder.setDataCallback(&player)->setSampleRate(out_sr)->setFramesPerCallback(audio_buffer_size);;

    oboe::ManagedStream instream, outstream;
    auto result = in_builder.openManagedStream(instream);
    if (result != oboe::Result::OK) {
        fprintf(stderr, "Failed to open input audio stream: %s\n", oboe::convertToText(result));
        return EXIT_FAILURE;
    }

    result = out_builder.openManagedStream(outstream);
    if (result != oboe::Result::OK) {
        fprintf(stderr, "Failed to open output audio stream: %s\n", oboe::convertToText(result));
        return EXIT_FAILURE;
    }

    auto in_buf_result = instream->setBufferSizeInFrames(audio_buffer_size);
    auto out_buf_result = outstream->setBufferSizeInFrames(audio_buffer_size);

    if (in_buf_result) {
        printf("Input buffer size: %d frames\n", in_buf_result.value());
    } else {
        fprintf(stderr, "Failed to set input buffer size: %s\n", oboe::convertToText(in_buf_result.error()));
        return 1;
    }
    if (out_buf_result) {
        printf("Output buffer size: %d frames\n", out_buf_result.value());
    } else {
        fprintf(stderr, "Failed to set output buffer size: %s\n", oboe::convertToText(out_buf_result.error()));
        return 1;
    }
    // Start recording and passing through
    result = instream->requestStart();
    if (result != oboe::Result::OK) {
        fprintf(stderr, "Failed to start stream: %s\n", oboe::convertToText(result));
        return 1;
    }
    result = outstream->requestStart();
    if (result != oboe::Result::OK) {
        fprintf(stderr, "Failed to start stream: %s\n", oboe::convertToText(result));
        return 1;
    }


    printf("Audio passing through ...\n");

    while (run)
    {
        /* code */
    }
    

    outstream->requestStop();
    outstream->close();
    instream->requestStop();
    instream->close();

    recorder.dump("input.npy");
    player.dump("output.npy");
    printf("Done.\n");

    return  EXIT_SUCCESS;
}