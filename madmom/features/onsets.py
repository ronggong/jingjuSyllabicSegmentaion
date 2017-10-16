# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains onset detection related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter

# from ..processors import (Processor, SequentialProcessor, ParallelProcessor,
#                           BufferProcessor)
from ..audio.signal import smooth as smooth_signal
from ..utils import combine_events

EPSILON = np.spacing(1)



# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    Parameters
    ----------
    activations : numpy array
        Activation function.
    threshold : float
        Threshold for peak-picking
    smooth : int or numpy array, optional
        Smooth the activation function with the kernel (size).
    pre_avg : int, optional
        Use `pre_avg` frames past information for moving average.
    post_avg : int, optional
        Use `post_avg` frames future information for moving average.
    pre_max : int, optional
        Use `pre_max` frames past information for moving maximum.
    post_max : int, optional
        Use `post_max` frames future information for moving maximum.

    Returns
    -------
    peak_idx : numpy array
        Indices of the detected peaks.

    See Also
    --------
    :func:`smooth`

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), set `pre_avg` and
    `post_avg` to 0.
    For peak picking of local maxima, set `pre_max` and  `post_max` to 1.
    For online peak picking, set all `post_` parameters to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    """
    # smooth activations
    activations = smooth_signal(activations, smooth)
    # compute a moving average
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # TODO: make the averaging function exchangeable (mean/median/etc.)
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_avg = uniform_filter(activations, filter_size, mode='constant',
                                 origin=avg_origin)
    else:
        # do not use a moving average
        mov_avg = 0
    # detections are those activations above the moving average + the threshold
    detections = activations * (activations >= mov_avg + threshold)
    # peak-picking
    max_length = pre_max + post_max + 1
    if max_length > 1:
        # compute a moving maximum
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_max = maximum_filter(detections, filter_size, mode='constant',
                                 origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError('`activations` must be either 1D or 2D')




class OnsetPeakPickingProcessor():
    """
    This class implements the onset peak-picking functionality.
    It transparently converts the chosen values from seconds to frames.

    Parameters
    ----------
    threshold : float
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one onset within `combine` seconds.
    delay : float, optional
        Report the detected onsets `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    onsets : numpy array
        Detected onsets [seconds].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    Examples
    --------
    Create a PeakPickingProcessor. The returned array represents the positions
    of the onsets in seconds, thus the expected sampling rate has to be given.

    # >>> proc = OnsetPeakPickingProcessor(fps=100)
    # >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.OnsetPeakPickingProcessor object at 0x...>

    Call this OnsetPeakPickingProcessor with the onset activation function from
    an RNNOnsetProcessor to obtain the onset positions.

    # >>> act = RNNOnsetProcessor()('tests/data/audio/sample.wav')
    # >>> proc(act)  # doctest: +ELLIPSIS
    array([ 0.09,  0.29,  0.45,  ...,  2.34,  2.49,  2.67])

    """
    FPS = 100
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.
    POST_AVG = 0.
    PRE_MAX = 0.
    POST_MAX = 0.
    COMBINE = 0.03
    DELAY = 0.
    ONLINE = False

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, online=ONLINE, fps=FPS,
                 **kwargs):
        # pylint: disable=unused-argument
        # TODO: make this an IOProcessor by defining input/output processings
        #       super(PeakPicking, self).__init__(peak_picking, write_events)
        #       adjust some params for online mode?
        if online:
            # set some parameters to 0 (i.e. no future information available)
            smooth = 0
            post_avg = 0
            post_max = 0
            # init buffer
            self.buffer = None
            self.counter = 0
            self.last_onset = None
        # save parameters
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.online = online
        self.fps = fps

    def reset(self):
        """Reset OnsetPeakPickingProcessor."""
        self.buffer = None
        self.counter = 0
        self.last_onset = None

    def process(self, activations, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        if self.online:
            return self.process_online(activations, **kwargs)
        else:
            return self.process_sequence(activations, **kwargs)

    def process_sequence(self, activations, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        # import time
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        # print(self.threshold)
        # time.sleep(5.5)  # pause 5.5 seconds

        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                            self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        onsets = peak_picking(activations, self.threshold, *timings)
        # convert to timestamps
        onsets = onsets.astype(np.float) / self.fps
        # shift if necessary
        if self.delay:
            onsets += self.delay
        # combine onsets
        if self.combine:
            onsets = combine_events(onsets, self.combine, 'left')
        # return the onsets
        return np.asarray(onsets)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        # buffer data
        if self.buffer is None or reset:
            # reset the processor
            self.reset()
            # put 0s in front (depending on conext given by pre_max
            init = np.zeros(int(np.round(self.pre_max * self.fps)))
            buffer = np.insert(activations, 0, init, axis=0)
            # offset the counter, because we buffer the activations
            self.counter = -len(init)
            # use the data for the buffer
            self.buffer = BufferProcessor(init=buffer)
        else:
            buffer = self.buffer(activations)
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                            self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        peaks = peak_picking(buffer, self.threshold, *timings)
        # convert to onset timings
        onsets = (self.counter + peaks) / float(self.fps)
        # increase counter
        self.counter += len(activations)
        # shift if necessary
        if self.delay:
            raise ValueError('delay not supported yet in online mode')
        # report only if there was no onset within the last combine seconds
        if self.combine and onsets.any():
            # prepend the last onset to be able to combine them correctly
            start = 0
            if self.last_onset is not None:
                onsets = np.append(self.last_onset, onsets)
                start = 1
            # combine the onsets
            onsets = combine_events(onsets, self.combine, 'left')
            # use only if the last onsets differ
            if onsets[-1] != self.last_onset:
                self.last_onset = onsets[-1]
                # remove the first onset if we added it previously
                onsets = onsets[start:]
            else:
                # don't report an onset
                onsets = np.empty(0)
        # return the onsets
        return onsets

    @staticmethod
    def add_arguments(parser, threshold=THRESHOLD, smooth=None, pre_avg=None,
                      post_avg=None, pre_max=None, post_max=None,
                      combine=COMBINE, delay=DELAY):
        """
        Add onset peak-picking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        threshold : float
            Threshold for peak-picking.
        smooth : float, optional
            Smooth the activation function over `smooth` seconds.
        pre_avg : float, optional
            Use `pre_avg` seconds past information for moving average.
        post_avg : float, optional
            Use `post_avg` seconds future information for moving average.
        pre_max : float, optional
            Use `pre_max` seconds past information for moving maximum.
        post_max : float, optional
            Use `post_max` seconds future information for moving maximum.
        combine : float, optional
            Only report one onset within `combine` seconds.
        delay : float, optional
            Report the detected onsets `delay` seconds delayed.

        Returns
        -------
        parser_group : argparse argument group
            Onset peak-picking argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add onset peak-picking related options to the existing parser
        g = parser.add_argument_group('peak-picking arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='detection threshold [default=%(default).2f]')
        if smooth is not None:
            g.add_argument('--smooth', action='store', type=float,
                           default=smooth,
                           help='smooth the activation function over N '
                                'seconds [default=%(default).2f]')
        if pre_avg is not None:
            g.add_argument('--pre_avg', action='store', type=float,
                           default=pre_avg,
                           help='build average over N previous seconds '
                                '[default=%(default).2f]')
        if post_avg is not None:
            g.add_argument('--post_avg', action='store', type=float,
                           default=post_avg, help='build average over N '
                           'following seconds [default=%(default).2f]')
        if pre_max is not None:
            g.add_argument('--pre_max', action='store', type=float,
                           default=pre_max,
                           help='search maximum over N previous seconds '
                                '[default=%(default).2f]')
        if post_max is not None:
            g.add_argument('--post_max', action='store', type=float,
                           default=post_max,
                           help='search maximum over N following seconds '
                                '[default=%(default).2f]')
        if combine is not None:
            g.add_argument('--combine', action='store', type=float,
                           default=combine,
                           help='combine events within N seconds '
                                '[default=%(default).2f]')
        if delay is not None:
            g.add_argument('--delay', action='store', type=float,
                           default=delay,
                           help='report the events N seconds delayed '
                                '[default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g