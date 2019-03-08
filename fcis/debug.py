import mxnet as mx

# debug 1
x = mx.nd.array([[[[0.,   1.,   2.,   3.,   4.,   5.],
                   [6.,   7.,   8.,   9.,  10.,  11.],
                   [12.,  13.,  14.,  15.,  16.,  17.],
                   [18.,  19.,  20.,  21.,  22.,  23.],
                   [24.,  25.,  26.,  27.,  28.,  29.],
                   [30.,  31.,  32.,  33.,  34.,  35.],
                   [36.,  37.,  38.,  39.,  40.,  41.],
                   [42.,  43.,  44.,  45.,  46.,  47.]]]],
                mx.gpu(1))

y = mx.nd.array([[0, 0, 0, 4, 4]],
                mx.gpu(1))

conv = mx.sym.Variable(name='conv')
roi = mx.sym.Variable(name='roi')

sym_roi_pooling = mx.sym.ROIPooling(data=conv, rois=roi, pooled_size=(2,2), spatial_scale=1, name='roi_pooling')

sym_roi_align = mx.sym.ROIAlign(data=conv, rois=roi, pooled_size=(2,2), spatial_scale=1, name='roi_align')

executor_roi_pooling = sym_roi_pooling.bind(ctx=mx.gpu(1), args={'conv': x, 'roi': y})
executor_roi_pooling.forward()

executor_roi_align = sym_roi_align.bind(ctx=mx.gpu(1), args={'conv': x, 'roi': y})
executor_roi_align.forward()

print executor_roi_pooling.outputs[0].asnumpy()
print executor_roi_align.outputs[0].asnumpy()


# debug 2
data = mx.symbol.Variable(name='data')
rois = mx.symbol.Variable(name='rois')

ra = mx.symbol.ROIAlign(data=data, rois=rois, pooled_size=(2, 2), spatial_scale=1)
rp = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(2, 2), spatial_scale=1)

h, w = 10, 10
x = [[[[i*w+j for j in range(w)] for i in range(h)]]]
y = [[0, 0, 0, 2, 2]]

x = mx.nd.array(x, dtype='float32',ctx=mx.gpu(1))
y = mx.nd.array(y, dtype='float32',ctx=mx.gpu(1))

ex_ra = ra.bind(ctx=mx.gpu(1), args={'data':x, 'rois':y})
ex_rp = rp.bind(ctx=mx.gpu(1), args={'data':x, 'rois':y})

print(ex_ra.forward()[0].asnumpy())
print(ex_rp.forward()[0].asnumpy())
