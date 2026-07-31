The csv contain the fits to the shifts in the JP QNM frequencies as a function of the rotation parameter \chi. There are
seperate fits for the real and imaginary parts, as well as for even and odd parity modes. Note that odd parity modes
should be specified with a negative l value. These fits are for the shifts ONLY, so you still need to add the Kerr QNMs
back on. The fits take the form (VALID FOR \chi\in[0,.99]):

\delta\omega(l,m,n,\chi,Even/Odd,Re/Im)=\sum_{n=0}^{9} a_n log(1-\chi^2)^n

In general, these fits are within about 1% of the actual value in the applicable regime. Better fits are possible,
they're just not really necessary given the quality of data.