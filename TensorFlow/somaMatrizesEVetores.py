import tensorflow as tf

#soma de vetores
vetor1 = tf.constant([1,2,3])
vetor2 = tf.constant([3,4,5])

soma = vetor1 + vetor2


with tf.compat.v1.Session() as sess:
    print(sess.run(soma))
    
#soma de matriz dimensionais
m1 = tf.constant([[1,2,3],[4,5,6,]])
m2 = tf.constant([[1,2,3],[4,5,6,]])

soma2 = tf.add(m1,m2)

with tf.compat.v1.Session() as sess:
    print(sess.run(soma2))
    

#soma de colunas
m3 = tf.constant([[1,2,3],[4,5,6,]])
m4 = tf.constant([[1],[2]])

soma3 = tf.add(m3,m4)

with tf.compat.v1.Session() as sess:
    print(sess.run(soma3))
    