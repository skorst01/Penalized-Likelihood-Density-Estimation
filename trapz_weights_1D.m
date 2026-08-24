function w = trapz_weights_1D(x)

x = x(:);
n = length(x);

w = zeros(n,1);
w(1) = (x(2)-x(1))/2;
w(end) = (x(end)-x(end-1))/2;

for i = 2:n-1
    w(i) = (x(i+1)-x(i-1))/2;
end

end