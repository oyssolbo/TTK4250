# set to false for speedup.
# Speedup will also occur if argument '-O' is given to python,
# as __debug__ then is False
DEBUG = False and __debug__


sigma_a = 26  # acceleration standard deviation
sigma_z = 3.1  # measurement standard deviation

# Clutter density, (measurements per m^2, is this reasonable?)
clutter_density = 1e-6

# Detection probability, (how often cyan dot appear, is this reasonable?)
detection_prob = 0.5

# Gate percentile, (estimated percentage of correct measurements that will be
# accepted by gate function)
gate_percentile = 0.8


"""
Been raging a bit too hard on python and numpy, so
my bloodpressure is a bit too high right now...

Perfect time to tune in other words!

I will try to rather get a feeling about the response, instead
of having a filter that is "perfect"....


Once my blood starts cooling down, pherhaps I try to get a more
optimized filter, but cannot guarantee that!


sigma_a and sigma_z is related to the EKF

sigma_z = R and is the noise belonging to the measurements. By 
increasing this, we expect that the measurements will have a 
larger noise, and thus trust the model more. For the PDA, this 
means that values that are further away from the measurement is
more related to noise in measurements instead of the actual
process. Increasing this should in fact reduce number of samples
that will actively be considered

sigma_z = Q is the noise related to the model. By increasing
this, we say that the process can change rapidly. This allows
us to increase the likelihood for measurements further away
from the priori estimate. Thus, considering more estimates

Clutter density is number of measruerments per m^2. Increasing
this allows the system to generate more clutter. I cannot identify
if this exact value is reasonable or not, however the clutter
density will vary depending on the accuracy of the system as well
as of the desired scan-velocity.
For example would a radar have lower clutter intensity with a lower
rotation speed, with the downside that it updates relatively 
infrequently. Increasing the rotation speed allwos it to scan the
area quicker, however will allow less time for the sensor cells 
to acquire and identify a possible signal.

Detection probability should be in the range [0.5, 1)
Lower bounded by 0.5 since that value means that we are just
guessing yes/no if a detection has been made.


Tuning:

Attempt 0:
    Values: 
        sigma_a = 2.6
        sigma_z = 3.1
        clutter_density = 1e-6
        detection_prob = 0.5
        gate_percentile = 0.8
    Result:
        36.9% within 95% confidence
        0.3% below
        62.8% above

Attempt 1:
    Values: 
        sigma_a = 2.6
        sigma_z = 31
        clutter_density = 1e-6
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Tries to increase sigma_z to check 
        if that gives further restrictions
        in number of measurements we consider.
        Expected that this will limit the 
        gate that is used  
    Result:
        Can clearly see that the gate has been
        increased, so I was wrong. I expected the
        system to favorize measurments closer
        to the expected value, however I did not
        take into account that the gated measurements
        must be increased to offset that the 
        measurements will be less accurate

        91.9% within 95% confidence
        3.1% below
        5.8% above

Attempt 1:
    Values: 
        sigma_a = 26
        sigma_z = 3.1
        clutter_density = 1e-6
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Tries to increase sigma_a to check if an
        increased value gives us further values 
        that could be evaluated. Expectation say
        yes, as larger deviations in the process model
        allows the model/system to change much between
        measurmeents
    Result:
        Can clearly see that the gate has been
        increased

        91.9% within 95% confidence
        3.1% below
        5.8% above





"""
