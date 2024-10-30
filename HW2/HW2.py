# Alex Welk - CMSC438 - HW2
import math

def dotting(f):
    var = []
    dotted = []

    # ensures f is interable even when its a single instruction 
    if isinstance(f, tuple) and isinstance(f[0], str):
        f = [f]

    for i, instruction in enumerate(f):
        match instruction[0]:
            case 'X': # X input
                x = instruction[1]
                var.append(x)
                dotted.append(1.0)
            case 'C': # Constant input
                c = instruction[1]
                var.append(c)
                dotted.append(0.0) # Derivative of constant
            case '+': #Addition
                u = instruction[1]
                v = instruction[2]
                var.append(var[u] + var[v])
                dotted.append(dotted[u] + dotted[v])
            case '-': # Substraction
                u = instruction[1]
                v = instruction[2]
                var.append(var[u] - var[v])
                dotted.append(dotted[u] - dotted[v])
            case "*": # Multiplication
                u = instruction[1]
                v = instruction[2]
                var.append(var[u] * var[v])
                dotted.append(dotted[u] * var[v] + dotted[v] * var[u])
            case '/': #Division
                u = instruction[1]
                v = instruction[2]
                var.append(var[u] / var[v])
                dotted.append((dotted[u] * var[v] - var[u] * dotted[v]) / (var[v]**2))
            case 'S': #sqaure
                u = instruction[1]
                var.append(var[u] ** 2)
                dotted.append(2.0 * var[u] * dotted[u])
            case 'E': # Exponent
                u =  instruction[1]
                var.append(math.exp(var[u]))
                dotted.append(math.exp(var[u]) * dotted[u])
            case 'L': # Log
                u = instruction[1]
                var.append(math.log(var[u]))
                dotted.append(dotted[u] / var[u]) 


    y = var[-1]  # Final value of the function
    dotted_y = dotted[-1]  # Final dotted value


    return x, y, dotted_y


f_specification = (
('X', 2.0),
('C', 1.0),
('+', 0, 1),
('*', 0, 2),
('S', 3)
)
x, y, dotted_y = dotting(f_specification)
print(x, y, dotted_y)

f_specification = (('X', 3.0), ('C', 2.0), ('-', 1, 0))
x, y, dotted_y = dotting(f_specification)
print(x, y, dotted_y)
