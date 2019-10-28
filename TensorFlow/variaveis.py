import tensorflow as tf

valor1 = tf.constant(15, name='valor1')
print(valor1)

soma = tf.Variable(valor1 + 5, name='valor1')
print(soma)
type(soma)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(soma))