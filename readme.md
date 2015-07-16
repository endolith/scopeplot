Oscilloscope-like plotting of signals, showing density of waveforms instead 
of just the peak amplitude silhouettes.

Vertical cross section through this plot is a histogram of the waveform values 
for that chunk as brightness.

Input signal must have been sampled at twice the highest frequency present in
the signal, like audio.

Signal is first FFT interpolated to get inter-sample information, then broken
up into overlapping chunks, 1 for each pixel, then those are linear
interpolated, with each line segment contributing to 1 or 2 pixels, depending
on where it occurs in the chunk.

TODO:

- Handle line segments that go outside the visible range
- Handle circularity/end behavior as a parameter
 - Fix white dots at endpoints
- Read files one chunk at a time, FFT resample each chunk to memory errors
- Show original samples as circular dots when zoomed in enough
- Show RMS value (window parameter?)
- Show (intersample) peak value
- Color the waveform based on spectral centroid, spectral content, etc.
- Use randomized resampling?  Completely different, though.

Related: https://github.com/endolith/freesound-thumbnailer