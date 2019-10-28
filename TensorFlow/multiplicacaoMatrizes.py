import tensorflow as tf

#multiplicacao normal linhas x colunas
m1 = tf.constant([[1,2],[3,4]])
m2 = tf.constant([[-1,3],[4,2]])

mul = tf.matmul(m1,m2)

with tf.compat.v1.Session() as sess:
    print(sess.run(mul))
    
    

#multiplicar AXB É DIFERENTE DE BXA
    
    
    
#multiplicacao primeira linha x  primeira coluna
a1 = tf.constant([[2,3],[0,1],[-1,4]])
b1 = tf.constant([[1,2,3],[-2,0,4]])

mul2 = tf.matmul(a1,b1)

with tf.compat.v1.Session() as sess:
    print(sess.run(a1))
    print("\n")
    print(sess.run(b1))
    print("\n")
    print(sess.run(mul2))
    
    
