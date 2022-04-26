numpy中array，asarray和asanyarray的区别

一、先讨论默认情况下
1. array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会。
也就是说当数据=是ndarray时，a = array（b），a和b不再是占用同一个内存的数组，而asarray中，a和b是同一个，改变b即改变a。
2. asarray和asanyarray的区别，np.asanyarray 会返回 ndarray 或者ndarray的子类，而np.asarray 只返回 ndarray. 也就是说对于ndarray的子类，asanyarray是不会复制的。

二、array、asarray，asanyarray的区别还受到两个参数控制 即copy和subok

array默认设置copy=True

假设a是一个数组，m是一个矩阵，它们的数据类型都是float32：

np.array（a）和np.array（m）都将复制，因为这是默认行为。

np.array（a，copy=False）和np.array（m，copy=False）将复制m而不是a，因为m不是ndarray。

np.array（a，copy=False，subok=True）和np.array（m，copy=False，subok=True）都不会复制，因为m是矩阵，它是ndarray的子类。

由于数据类型不兼容，数组（a，dtype=int，copy=False，subok=True）将同时复制两者。

asanyarray：如果输入是兼容的ndarray或类似matrix的子类（copy=False，subok=True），则将返回未复制的输入。
