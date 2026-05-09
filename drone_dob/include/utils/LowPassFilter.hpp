#ifndef LOW_PASS_FILTER_HPP
#define LOW_PASS_FILTER_HPP

#include <iostream>
#include <cmath>

class LowPassFilter{

    public:

    // Default Constructors
    LowPassFilter();

    // Parameterized Constructor
    LowPassFilter(double cutoff_freq);

    // Method to set cutoff frequency
    void setCutoffFrequency(double cutoff_freq);

    // Method to update the low-pass filter
    double update(double input, double dt);

    double getOutput() const;

    private:

    double cutoff_frequency_; // Cutoff frequency
    double output_{0.0};      // Filtered output

};


#endif // LOW_PASS_FILTER_HPP