import tensorflow as tf

v1 = tf.constant(5)
v2 = tf.constant(7)
type(v1)

soma = v1 + v2
type(soma)

print(soma)

#sempre precisa de uma sessão para execução
with tf.compat.v1.Session() as sess:
    s = sess.run(soma)
    
print(s)