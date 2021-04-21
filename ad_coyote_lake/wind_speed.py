import scipy
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt


def wind_speeds_from_normal(mu, sigma, number, cutoff, plot=False, normalize=True):
    """
    Generates a respresentative sample of wind speeds (and weights) from normal
    distribution with mean mu and standard deviation sigma. 

    Arguments:
      mu        = mean of 
      mu        = mean of normal distribution
      sigma     = standard deviation of normal distribution 
      number    = number of sample points
      cutoff    = cutoff probability for seletecting end points of samples
      plot      = flag indicating whether or not to plot the results
      normalize = flag indicating whether or not to normalize weight to ensure they
                  sum to 1.

    Return: tuple (speed, probs) where
      speed      = array of sampled wind speeds
      probs      = probability for wind speed bin 

    """

    delta = sigma*scipy.sqrt(2.0)*scipy.special.erfinv(cutoff)
    x_max = mu + delta
    x_min = mu - delta
    x_vals = scipy.linspace(x_min, x_max, number)
    f_vals = normal_pdf(x_vals, mu, sigma)
    dx = x_vals[1] - x_vals[0]
    if normalize:
        # normalize to ensure sum to 1  
        total = f_vals.sum()*dx
        probs = f_vals*dx/total
    else:
        probs = f_vals*dx

    if plot:
        x_fine = scipy.linspace(x_min, x_max,100*number)
        f_fine = normal_pdf(x_fine, mu, sigma)
        plt.plot(x_fine, f_fine,'b')
        plt.plot(x_vals, f_vals,'go')
        plt.grid(True)
        plt.xlabel('speed (m/s)')
        plt.ylabel('prob. density')
        #plt.show()
    return x_vals, probs


def wind_speeds_from_file(filename, number, start_frac=0.0, stop_frac=1.0, plot=False, normalize=True):
    """ 
    Generates a representative sample of wind speeds (and weights) from data
    file (filename) containing a numerical approximation of wind speed pdf.
    Returns a list of winds speeds at weight which are sampled from the pdf at
    an evenly space number of sample points. 

    Argugments:
      filename   = name of wind speed data file
      number     = number of sample points
      start_frac = fraction between min and max to start even sampling
      stop_frac  = fraction between min and max to stop even sampling
      plot       = flag indicating whether or not to plot the results
      normalize  = flag indicating whether or not to normalize weight to ensure they
                   sum to 1.

    Return: tuple (speed, probs) where
      speed      = array of sampled wind speeds
      probs      = probability for wind speed bin 

    """
    #Load data and get interpolation function and limits.
    wind_speed_data = scipy.loadtxt(filename)
    wind_speed_func = get_interp_func(wind_speed_data)
    min_speed, max_speed = get_limits(wind_speed_data)
    rng_speed = max_speed - min_speed

    # Interpolate data
    s0 = min_speed + start_frac*rng_speed
    s1 = min_speed + stop_frac*rng_speed
    speed = scipy.linspace(s0, s1, number)
    density = wind_speed_func(speed)

    # Get probabilities 
    ds = speed[1] - speed[0]
    if normalize:
        # normalize to ensure sum to 1  
        total = density.sum()*ds
        probs = density*ds/total
    else:
        probs = density*ds

    if plot:
        plt.plot(wind_speed_data[:,0], wind_speed_data[:,1],'b')
        plt.plot(speed, density,'go')
        plt.grid(True)
        plt.xlabel('speed (m/s)')
        plt.ylabel('prob. density')
        #plt.show()
    return speed, probs


def normal_pdf(x, mu, sigma): 
    """
    Normal distribution pdf function with mean my and standard deviation sigma. 

    Arguments:
      x      = point (or array of points) at which to evaluate function
      mu     = mean of normal distribution
      sigma  = standard deviation of normal distribution 

    Return: 
      value  = value of normal distribution pdf at point (or array of points) x

    """
    value = (1.0/(sigma*scipy.sqrt(2*scipy.pi)))*scipy.exp(-(x-mu)**2/(2*sigma**2))
    return value 


def get_limits(data, col=0):
    """ 
    Get limits for specified column of data array. 
    """
    return data[:,col].min(), data[:,col].max()


def get_interp_func(data):
    """
    Returns function which interpolates data array. 
    """
    return scipy.interpolate.interp1d(data[:,0],data[:,1])


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    if 1:
        filename = 'wind_speed_pdf.txt'
        speed, probs = wind_speeds_from_file(filename, 30, 0.05, 0.85, plot=True)

    if 1:
        speed, probs = wind_speeds_from_normal(0.8, 0.1, 51, 0.99, plot=True, normalize=False)

    print()
    print('probs.sum() = {}'.format(probs.sum()))

    plt.show()


