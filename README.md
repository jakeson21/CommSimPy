# CommSimPy
Python based Communication System utilities

``` python
import Utils
import matplotlib.pyplot as plt

L=128
Fs = 2
fgp = Utils.read_point_data_file('example_filter.csv')
imp, h_new = Utils.fgp_to_impulse_response(fgp, fs=Fs, length=L, gdelay=L/2.)
plt.plot(h_new[:,0]*200, h_new[:,1])
plt.show()

plt.plot(imp)
plt.show()
np.array2string(imp.flatten(), separator=',', max_line_width=9999)
```
