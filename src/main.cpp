#include <App.hpp>

std::atomic<bool> run = true;
void sigint_handler(int arg)
{
    run = false;
}

int main(int argc, char ** argv)
{
    signal(SIGINT, sigint_handler);
    cxxopts::Options options { "FilterProgram", "Audio passing through filter program" };

    options.add_options()
    ("h,help", "Print usage")
    ("m,model",        "File containing the model to load (expected .onnx file)", cxxopts::value<std::string>()->default_value("./lowpass_rnn.onnx"))
    ("f,fc",        "Cutoff frequency (Hz)", cxxopts::value<int32_t>())
    ("p,profiling", "Profiling mode : get information about session perfomance (boolean)", cxxopts::value<bool>()->default_value("false"))
    ("r,run_duration", "Run duration (seconds): indicate of much time to run the program (if not specified, the program runs until stopped with Ctrl+C)", cxxopts::value<int>())
    ("d,debug", "Debug mode : get session input and output signals (boolean)", cxxopts::value<bool>()->default_value("false"))
    ("c,cpu_only", "CPU only mode : NNAPI will not try to run inference on GPU/NPU (boolean)", cxxopts::value<bool>()->default_value("false"))
    ;

    auto args = options.parse(argc, argv);
    
    if (args.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    App app(args, run);
    app.run();
    return  EXIT_SUCCESS;
}
