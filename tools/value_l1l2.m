function [norm_value] = value_l1l2(W)
n = size(W,2);
norm_value = 0;
for i=1:n
    tmp_vector = W(:,i);
    norm_value = norm_value + norm(tmp_vector);
end
end