V_node está ordenada según:

V nodos con fuentes de tensión
V de nodos con cargas (corrientes conocidas)
V de nodos de transición (corrientes nulas)

A su vez cada uno de estos bloques se ordenan según orden de aparición en el archivo .json


V_known tiene los nodos con fuentes de tensión
V_unknown tiene los demás (nodos de corrientes conocidas, cargas y transición)

V_sorted está ordenada de manera tal que los nodos que pertenecen al mismo bus estén juntos y en orden.
A su vez el orden de los buses está según |buses| del .json

