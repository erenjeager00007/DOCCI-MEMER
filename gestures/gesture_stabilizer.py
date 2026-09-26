""" 
DOCCI MEMER - Gesture Stabilizer 
 
A tiny helper that smooths out noisy frame-by-frame gesture detections. 
 
Face landmark detection can flicker between gestures from one frame 
to the next (e.g. NEUTRAL -> SMILE -> NEUTRAL within a few frames). 
GestureStabilizer only "accepts" a new gesture once it has been seen 
for a number of consecutive frames in a row, which filters out that 
kind of flicker. 
 
No external libraries are used - just plain Python. 
""" 
 
 
class GestureStabilizer: 
    """ 
    Tracks incoming gesture strings and only reports a gesture as 
    "stable" once it has appeared for `stable_frames` frames in a row. 
    """ 
 
    def __init__(self, stable_frames=5): 
        """ 
        Parameters 
        ---------- 
        stable_frames : int 
            How many consecutive identical frames are required before 
            a gesture is considered stable. Defaults to 5. 
        """ 
        self.stable_frames = stable_frames 
 
        # The gesture we are currently counting consecutive frames for. 
        self._current_gesture = None 
 
        # How many consecutive frames _current_gesture has been seen. 
        self._consecutive_count = 0 
 
        # The last gesture that was confirmed as "stable". 
        # Starts as NEUTRAL so callers always get a sensible default. 
        self._stable_gesture = "NEUTRAL" 
 
    def update(self, gesture): 
        """ 
        Feed in the latest detected gesture for this frame. 
 
        Parameters 
        ---------- 
        gesture : str 
            The gesture detected this frame, e.g. "SMILE" or "NEUTRAL". 
 
        Returns 
        ------- 
        str 
            The currently stable gesture (unchanged until a new 
            gesture has been seen consecutively `stable_frames` times). 
        """ 
 
        # If the gesture matches what we were already counting, 
        # increase the streak count. 
        if gesture == self._current_gesture: 
            self._consecutive_count += 1 
        else: 
            # The gesture changed - reset and start counting this new one. 
            self._current_gesture = gesture 
            self._consecutive_count = 1 
 
        # Once the streak is long enough, accept it as the stable gesture. 
        if self._consecutive_count >= self.stable_frames: 
            self._stable_gesture = self._current_gesture 
 
        return self._stable_gesture 
