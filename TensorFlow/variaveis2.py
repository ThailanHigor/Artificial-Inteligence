import tensorflow as tf

vetor = tf.constant([5, 15, 20], name='vetor')
type(vetor)

soma = tf.Variable(vetor + 5, name='soma')
print(soma)
type(soma)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(soma))
    
    
valor = tf.Variable(0, name='valor')
init2 = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init2)
    for i in range(5):
        valor = valor + 1
        print(sess.run(valor))