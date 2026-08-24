function [ML_pen_val, grad] = ML_pen_fun_grad( ...
    coefs, Zx, Zy, cx, dy, n, m, N, ...
    xq, yq, Zxq, Zyq, ...
    Pxkron, Pykron, Px, Py, rho_x, rho_y)

% =========================================================
% ROZKLAD KOEFICIENTU
% =========================================================
z = coefs(1:n*m);
v = coefs(n*m+1 : n*m+n);
u = coefs(n*m+n+1 : end);

Z = reshape(z, n, m);

% =========================================================
% DATOVY CLEN
% l1 = tr(Zx * Z * Zy') + cx'v + dy'u
% =========================================================
data_int = sum(sum((Zx * Z) .* Zy));

ML_data = -data_int - cx' * v - dy' * u;

% =========================================================
% INTEGRACNI CLEN
% N * log int int exp(S)
% =========================================================
sp_int = Zxq * Z * Zyq';
sp_x   = Zxq * v;
sp_y   = Zyq * u;

S = sp_int + sp_x + sp_y';

Smax = max(S(:));
E = exp(S - Smax);

wx = trapz_weights(xq);
wy = trapz_weights(yq);

W = E .* (wx(:) * wy(:)');

A_stab = sum(W(:));
logA = Smax + log(A_stab);

ML_val = ML_data + N * logA;

% =========================================================
% PENALIZACE
% =========================================================
pen = rho_x * (z' * Pxkron * z + v' * Px * v) ...
    + rho_y * (z' * Pykron * z + u' * Py * u);

ML_pen_val = ML_val + pen;

% =========================================================
% GRADIENT DATOVEHO CLENU
% =========================================================
grad_Z_data = -Zx' * Zy;
grad_z_data = grad_Z_data(:);

grad_v_data = -cx;
grad_u_data = -dy;

% =========================================================
% GRADIENT INTEGRACNIHO CLENU
% =========================================================
GZ_int = Zxq' * W * Zyq;
gv_int = Zxq' * sum(W, 2);
gu_int = Zyq' * sum(W, 1)';

grad_z_int = N * GZ_int(:) / A_stab;
grad_v_int = N * gv_int / A_stab;
grad_u_int = N * gu_int / A_stab;

% =========================================================
% GRADIENT PENALIZACE
% =========================================================
grad_z_pen = 2 * rho_x * Pxkron * z + 2 * rho_y * Pykron * z;
grad_v_pen = 2 * rho_x * Px * v;
grad_u_pen = 2 * rho_y * Py * u;

% =========================================================
% CELKOVY GRADIENT
% =========================================================
grad_z = grad_z_data + grad_z_int + grad_z_pen;
grad_v = grad_v_data + grad_v_int + grad_v_pen;
grad_u = grad_u_data + grad_u_int + grad_u_pen;

grad = [grad_z; grad_v; grad_u];

end