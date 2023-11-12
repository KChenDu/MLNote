1. [卷积运算](convolutional_layers.ipynb)
2. 动机
    - 稀疏交互：这是使核的大小远小于输入的大小来达到的。在深度卷积网络中，处在网络深层的单元可能与绝大部分输入是间接交互的。这允许网络可以通过只描述稀疏交互的基石来高效地描述多个变量的复杂交互
	    ```mermaid
		graph LR
			subgraph 稀疏连接
			x11((x1))
			x12((x2))
			x13((x3))
			x14((x4))
			x15((x5))
			s11((s1))
			s12((s2))
			s13((s3))
			s14((s4))
			s15((s5))
			
			x11 --> s11
			x11 --> s12
			x12 --> s11
			x12 --> s12
			x12 --> s13
			x13 --> s12
			x13 --> s13
			x13 --> s14
			x14 --> s13
			x14 --> s14
			x14 --> s15
			x15 --> s14
			x15 --> s15
			end
			subgraph 全连接
			x21((x1))
			x22((x2))
			x23((x3))
			x24((x4))
			x25((x5))
			s21((s1))
			s22((s2))
			s23((s3))
			s24((s4))
			s25((s5))
			
			x21 --> s21
			x21 --> s22
			x21 --> s23
			x21 --> s24
			x21 --> s25
			x22 --> s21
			x22 --> s22
			x22 --> s23
			x22 --> s24
			x22 --> s25
			x23 --> s21
			x23 --> s22
			x23 --> s23
			x23 --> s24
			x23 --> s25
			x24 --> s21
			x24 --> s22
			x24 --> s23
			x24 --> s24
			x24 --> s25
			x25 --> s21
			x25 --> s22
			x25 --> s23
			x25 --> s24
			x25 --> s25
			end
			style x13 fill:#808080
			style s12 fill:#808080
			style s13 fill:#808080
			style s14 fill:#808080
			style x23 fill:#808080
			style s21 fill:#808080
			style s22 fill:#808080
			style s23 fill:#808080
			style s24 fill:#808080
			style s25 fill:#808080
		```
        ```mermaid
		graph LR
			subgraph 稀疏连接
			x11((x1))
			x12((x2))
			x13((x3))
			x14((x4))
			x15((x5))
			s11((s1))
			s12((s2))
			s13((s3))
			s14((s4))
			s15((s5))
			
			x11 --> s11
			x11 --> s12
			x12 --> s11
			x12 --> s12
			x12 --> s13
			x13 --> s12
			x13 --> s13
			x13 --> s14
			x14 --> s13
			x14 --> s14
			x14 --> s15
			x15 --> s14
			x15 --> s15
			end
			subgraph 全连接
			x21((x1))
			x22((x2))
			x23((x3))
			x24((x4))
			x25((x5))
			s21((s1))
			s22((s2))
			s23((s3))
			s24((s4))
			s25((s5))
			
			x21 --> s21
			x21 --> s22
			x21 --> s23
			x21 --> s24
			x21 --> s25
			x22 --> s21
			x22 --> s22
			x22 --> s23
			x22 --> s24
			x22 --> s25
			x23 --> s21
			x23 --> s22
			x23 --> s23
			x23 --> s24
			x23 --> s25
			x24 --> s21
			x24 --> s22
			x24 --> s23
			x24 --> s24
			x24 --> s25
			x25 --> s21
			x25 --> s22
			x25 --> s23
			x25 --> s24
			x25 --> s25
			end
			style x12 fill:#808080
			style x13 fill:#808080
			style x14 fill:#808080
			style s13 fill:#808080
			style x21 fill:#808080
			style x22 fill:#808080
			style x23 fill:#808080
			style x24 fill:#808080
			style x25 fill:#808080
			style s23 fill:#808080
		```
        ```mermaid
		graph TD
        subgraph 间接交互
            x1((x1))
            x2((x2))
            x3((x3))
            x4((x4))
            x5((x5))
            h1((h1))
            h2((h2))
            h3((h3))
            h4((h4))
            h5((h5))
            g1((g1))
            g2((g2))
            g3((g3))
            g4((g4))
            g5((g5))

            x1 --> h1
            x1 --> h2
            x2 --> h1
            x2 --> h2
            x2 --> h3
            x3 --> h2
            x3 --> h3
            x3 --> h4
            x4 --> h3
            x4 --> h4
            x4 --> h5
            x5 --> h4
            x5 --> h5
            h1 --> g1
            h1 --> g2
            h2 --> g1
            h2 --> g2
            h2 --> g3
            h3 --> g2
            h3 --> g3
            h3 --> g4
            h4 --> g3
            h4 --> g4
            h4 --> g5
            h5 --> g4
            h5 --> g5
        end
        style x1 fill:#808080
        style x2 fill:#808080
        style x3 fill:#808080
        style x4 fill:#808080
        style x5 fill:#808080
        style h2 fill:#808080
        style h3 fill:#808080
        style h4 fill:#808080
        style g3 fill:#808080
        ```
    - 参数共享：是指在一个模型的多个函数中使用相同的参数
        ```mermaid
		graph LR
			subgraph 参数共享
			x11((x1))
			x12((x2))
			x13((x3))
			x14((x4))
			x15((x5))
			s11((s1))
			s12((s2))
			s13((s3))
			s14((s4))
			s15((s5))
			
			x11 --> s11
			x11 --> s12
			x12 --> s11
			x12 --> s12
			x12 --> s13
			x13 --> s12
			x13 --> s13
			x13 --> s14
			x14 --> s13
			x14 --> s14
			x14 --> s15
			x15 --> s14
			x15 --> s15
			end
			subgraph 全连接
			x21((x1))
			x22((x2))
			x23((x3))
			x24((x4))
			x25((x5))
			s21((s1))
			s22((s2))
			s23((s3))
			s24((s4))
			s25((s5))
			
			x21 --> s21
			x21 --> s22
			x21 --> s23
			x21 --> s24
			x21 --> s25
			x22 --> s21
			x22 --> s22
			x22 --> s23
			x22 --> s24
			x22 --> s25
			x23 --> s21
			x23 --> s22
			x23 --> s23
			x23 --> s24
			x23 --> s25
			x24 --> s21
			x24 --> s22
			x24 --> s23
			x24 --> s24
			x24 --> s25
			x25 --> s21
			x25 --> s22
			x25 --> s23
			x25 --> s24
			x25 --> s25
			end
            linkStyle 0 stroke:black
            linkStyle 3 stroke:black
            linkStyle 6 stroke:black
            linkStyle 9 stroke:black
            linkStyle 12 stroke:black
            linkStyle 25 stroke:black
		```
    - 等变表示：如果一个函数满足输入改变，输出也以同样的方式改变这一性质，我们就说它是等变的。特别地，如果函数$f(x)$与$g(x)$满足$f(g(x)) = g(f(x))$，我们就说$f(x)$对于变换$g$具有等变性。对于卷积来说，如果令$g$是输入的任意平移函数，那么卷积函数对于$g$具有等变性

[返回](readme.md)