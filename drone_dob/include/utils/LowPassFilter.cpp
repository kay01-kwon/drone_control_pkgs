#include "LowPassFilter.hpp"

LowPassFilter::LowPassFilter()
: cutoff_frequency_(50.0) 
{

}

LowPassFilter::LowPassFilter(double cutoff_freq, double dt)
: cutoff_frequency_(cutoff_freq)
{

}

void LowPassFilter::setCutoffFrequency(double cutoff_freq) {
    cutoff_frequency_ = cutoff_freq;
}

double LowPassFilter::update(double input, double dt) {
    double tau = 1.0 / (2 * M_PI * cutoff_frequency_);
    double alpha = 1 - exp(-dt / tau);
    output_ = output_*(1 - alpha) + input * alpha;
    return output_;
}

double LowPassFilter::getOutput() const {
    return output_;
}