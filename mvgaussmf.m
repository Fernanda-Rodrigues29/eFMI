function [ y ] = mvgaussmf(x, mu, Sinv)
y = exp(-0.5*(x-mu)*Sinv*(x-mu)');
