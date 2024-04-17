beta=0.96;
r=0.04;
a_n=50000;
y_n=5;
rho=0.92;
sigma=0.05;
sd=3;
a_max=10;
sims=10000;
small_positive=10^(-4);
gamma=3.5;  %%%BUG 1: gamma is not preallocated

[P,y]=rouwen(rho,0,sigma,y_n);      %%changed the function and organized
a_min=-exp(y(1))/r;

a=linspace(0,a_max,a_n)';
[A, Y]=ndgrid(a,y);        
w_orig=linspace(a_min+exp(y(1)),(1+r)*a_max+exp(y(end)),a_n)'; %%%BUG 2: a_max should be multiplied with (1+r)
w1 = A + Y;    

tol=10^(-9);
maxits=10^7;

w=repmat(w_orig,1,y_n);
c2=small_positive*ones(a_n,y_n);
c1=max(w1,c2);
dif=1;
count=1;

tic    
while count<maxits && dif>tol
    for ys=1:y_n
c2(:, ys) = (beta * (1 + r) * interp1(w_orig, a, w1, 'linear', 'extrap').^(-gamma)* P(ys, :)').^(-1 / gamma); %%%BUG3: undefined variables  
    end
    c2(c2<0)=small_positive;
    w=A/(1+r)+c2;
    dif=max(max(abs(c1-c2)));
    count=count+1;
    c1=c2;
end
toc

figure(1)
subplot(2,1,1)
for i=1:y_n
    plot(w1(:,i),interp1(real(w(:,i)),real(c1(:,i)),w1(:,i),'pchip','extrap'))
    %plot(w1(:,i),c1(:,i))
    hold on
end
hold off
legend('lowest','y2','steady state','y4','highest')
xlabel('Cash-on-Hand')
ylabel('Consumption')

subplot(2,1,2)
plot(w1(:,1),interp1(real(w(:,1)),real(c1(:,1)),w1(:,1),'pchip','extrap'))
legend('lowest income')
xlabel('Cash-on-Hand')
ylabel('Consumption')

figure(2)
subplot(2,1,1)
for i=1:y_n
    plot(a,interp1(real(w(:,i)),real(c1(:,i)),a,'pchip','extrap'))
    %plot(w1(:,i),c1(:,i))
    hold on
end
hold off
legend('lowest','y2','steady state','y4','highest')
xlabel('Assets')
ylabel('Consumption')

subplot(2,1,2)
plot(a,interp1(real(w(:,1)),real(c1(:,1)),a,'pchip','extrap'))
legend('lowest income')
xlabel('Assets')
ylabel('Consumption')

y_sim=simulate(dtmc(P),sims);
w_sim=0;

for t=1:sims
    c_sim(t)=interp1(real(w(:,y_sim(t))),real(c1(:,y_sim(t))),real(w_sim(t)),'spline',min(min(c1)));
    if t<sims
        w_sim(t+1)=(1+r)*(w_sim(t)-c_sim(t))+exp(y(y_sim(t+1)));
    end
end

figure(3)
subplot(3,1,1)
plot(sims/2+1:sims,exp(y(y_sim(sims/2+1:sims))),sims/2+1:sims,ones(sims/2,1))
xlabel('Time')
ylabel('Income')

subplot(3,1,2)
plot(sims/2+1:sims,real(c_sim(sims/2+1:sims)))
xlabel('Time')
ylabel('Consumption')

subplot(3,1,3)
plot(sims/2+1:sims,real(w_sim(sims/2+1:sims)))
xlabel('Time')
ylabel('Cash-On-Hand')

%%  Correlation

[clgm,lags] = xcorr(y_sim,c_sim,4);
figure;
plot(lags,real(clgm));
xlabel("lags")
ylabel("Correlation Vector")
title("Correlogram between Simulated Income and Consumption Series")
legend("Correlation at 4 lags")
