function [val, grad] = ML_1D_grad(z, c, N, Zq, xq, P, rho)

z = z(:);
c = c(:);

s = Zq * z;
smax = max(s);
e = exp(s - smax);

A = trapz(xq, e);
logA = smax + log(A);

val = -c' * z + N * logA + rho * (z' * P * z);

w = trapz_weights_1D(xq);
w = w(:);

We = w .* e;
grad_logA = (Zq' * We) / A;

grad = -c + N * grad_logA + 2 * rho * P * z;
grad = grad(:);

end