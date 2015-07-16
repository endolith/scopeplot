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
- Read files one chunk at a time, FFT resample each chunk to avoid memory errors
- Show original samples as circular dots when zoomed in enough
- Show RMS value (window parameter?)
- Show (intersample) peak value
- Color the waveform based on spectral centroid, spectral content, etc.
- Use randomized resampling?  Completely different, though.

Related: 

- https://github.com/endolith/freesound-thumbnailer  
- http://dsp.stackexchange.com/q/184/29

Examples:

Guitar pluck:

[![plot of guitar](https://farm1.staticflickr.com/306/19701397555_58444c1ee0_z.jpg)](https://flic.kr/p/w1WP7c)

Violins:

[![plot of violins](https://farm1.staticflickr.com/422/19737001541_09726ae0c5_z.jpg)](https://flic.kr/p/w56hW2)

Sine wave:

[![plot of sine](https://farm1.staticflickr.com/417/19201290270_a91a64774e_z.jpg)](https://flic.kr/p/vfKCCN)

Noise:

[![plot of noise](https://farm1.staticflickr.com/395/19112954693_58b3ea3532_z.jpg)](https://flic.kr/p/v7WTxB)