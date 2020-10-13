# PyTorch的 transpose、permute、view、reshape

https://www.jianshu.com/p/54f5ccba4057

- reshape 封装了 view，view根据规则有时还需要调用contiguous()
- permute().contiguous().view()相当于reshape
- permute() 和 tranpose() 比较相似，transpose是交换**两个**维度，permute()是交换**多个**维度。

