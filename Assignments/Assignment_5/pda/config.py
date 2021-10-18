# set to false for speedup.
# Speedup will also occur if argument '-O' is given to python,
# as __debug__ then is False
DEBUG = False and __debug__


sigma_a = 2.6  # acceleration standard deviation
sigma_z = 5  # measurement standard deviation

# Clutter density, (measurements per m^2, is this reasonable?)
clutter_density = 67.581

# Detection probability, (how often cyan dot appear, is this reasonable?)
detection_prob = 0.896

# Gate percentile, (estimated percentage of correct measurements that will be
# accepted by gate function)
gate_percentile = 0.5


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

Attempt 2:
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
        Compared to the results in the attempt number 1,
        the gate looks to be back to its size in attempt 
        number 0. 

        About the NEES, we can see that most of the 
        measurements are far above the desired region for 
        the NEES. This means that the EKF has too low values
        in the Q-matrix, such that it is overconfident in
        its state estimates

        1.9% within 95% confidence
        0.2% below
        97.9% above

Attempt 3:
    Values: 
        sigma_a = 2.6
        sigma_z = 3.1
        clutter_density = 1e-2
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Tries to increase the probability for clutter, to
        see the results it have on the filter and the 
        estimates. By increasing the clutter, we expect that
        there will be more data sets inside of the gate, such
        that the filter will assess more values for the next
        estimate. Thus leaving the next estimate to be worse
        than in earlier attempts (attempt 0)
    Result:
        CI had honestly excpected more clutter on the screen.
        Trying to increase the clutter even further

        21.6% within 95% confidence
        0.3% below
        78.1% above

Attempt 4:
    Values: 
        sigma_a = 26
        sigma_z = 3.1
        clutter_density = 1e2
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Stated above
    Result:
        Heh. Just now realizes that there is a playable
        function. We could see that a clutter intensity of 
        1e2, makes the target lose track after a few 
        iterations

        11.6% within 95% confidence
        0% below
        88.4% above

# Trying to use the trajectory visualization from now on #

Attempt 5:
    Values: 
        sigma_a = 2.6
        sigma_z = 3.1
        clutter_density = 1e-6
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Redoing the earlier measurmenets
    Result:
        One can clearly see that the flter is overconfident.
        It diverges at around 15 to 16 iterations, where the
        filter goes on to its own thing. 

        To reduce the probability of this to occur, a less
        confident filter is required. That means to increase
        the value for sigma_z more compared to the values for
        sigma_a

        5.8% within 95% confidence
        0% below
        94.2% above

Attempt 6:
    Values: 
        sigma_a = 2.6
        sigma_z = 5
        clutter_density = 1e-6
        detection_prob = 0.5
        gate_percentile = 0.8
    Reasoning:
        Increasing the value for Q such that the system should 
        become less confident, and trust the measurements more
    Result:
        Gotta love it when the system diverged even quicker.
        Now it occured at iteration 8.... Just fml

        i have been testing some values for the EKF off screen,
        and I saw that I got fairly "good" results whn sigma_z
        was increased to around 20 (rough testing), where it 
        only started diverging at iteration number 136

        The problem is that the NEES is not the best. Trying to
        test different values to reduce NEES, eg increasing the 
        value for sigma_z, gives a larger covariance-matrix. For 
        the system to not diverge as quickly, a severe reduction in
        the agte percentile was required (currently testing
        sigma_z = 150 and gate_percentile = 0.01). But the 
        estimate diverging relatively quickly. 
        
        Reducing the parameters for sigma_z down to 10, while
        maintaining the gate_percentile at 0.01, gives a NEES
        where over 95% is within the 95% certainty. Howver, the
        number of gated measurements are so small, such that the
        system quickly invalidates the real measurements 

And there I lost all motivation....




Note written couple of days later:

I should have tuned this a lot better. The difficult thing is to get 
the system to follow the correct state without diverging too 
rapidly. My experimentation showed that the system would either accept
too many measurements and therefore diverge, or that it would not value 
the real system's measurements and thus diverge. I did experience that the
algortihm was able to follow the track somewhat, however the filter became
overconfident and started diverging at iteration 136. However, when running 
the same value on my desktop (right when I am writing this shit), the system 
diverged at iteration number 7 to 8. Idk why my laptop and desktop gave
different results....


I honestly started by trying to find the values that made the system stay 
within a reasonable NEES, however that didn't quite work out.

After that, I tried to just tune after analysing the track... It went as well
as one could expect.  

In other words, fuck my life!


"""
