y = [-67 -48 6 8 14 16 23 24 28 29 41 49 56 60 75];

% EM algorithm
K = 2;
N = length(y);
% (1) Initialize parameters
mu = rand(1,K);
al = rand(1,K);
sg = rand(1,K)*10;
while (true)
	% (2) E step
	gm = zeros(N,K);
	for j = (1:N) 
		for k = (1:K)
			gm(j,k) = al(k)*normpdf(y(j),mu(k),sg(k));
		end
		gm(j,:) /= sum(gm(j,:));
	end
	% (3) M step
	mu2 = y*gm./sum(gm);
	sg2 = sum(gm.*((y-mu').^2)')./sum(gm);
	al2 = sum(gm)/N;
	% (4) Check convergence
	di = sqrt(sum((mu2-mu).^2)+sum((sg2-sg).^2)+sum((al2-al).^2));
	if di < 0.0001
		break
	end
	mu = mu2;
	sg = sg2;
	al = al2;
end