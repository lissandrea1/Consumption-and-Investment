%% No Borrowing

beta = 0.96;
gamma = 1.3;
r = 0.04;
sigma = 0.04;

%%% Income Grid Setup
Y_n = 5; % Number of income grid points
sd = Y_n / 2 - 0.5; % Max number of standard deviations away from steady state for the income process.

Y = linspace(0, sd * sigma, Y_n);     %Set min to 0

%%% Asset Grid Setup
a_n = 1000; % Number of asset grid points
a_max = 4 * exp(Y(Y_n)); % Arbitrarily high number for the max of assets

A = linspace(-exp(Y(1)) / r, a_max, a_n)'; % Asset Grid Discretization
%%% Transition Probability Matrix for Income States
P = eye(Y_n); % This is just an identity matrix for illustration.

%% Calculations that do not need to be repeated

%%%%%   We will calculate the period utility here for all possible choices
%%%%%   of assets next period at each gripoint.

% Preallocate for consumption choices and utility
utility = zeros(a_n, Y_n, a_n);         %   Note that this is a three dimensional matrix, which is possible in Matlab.
%BUG 3: utility

for ap = 1:a_n
for y = 1:Y_n
for a = 1:a_n
c = (1 + r) * A(a) + exp(Y(y)) - A(ap);
    if c <= 0
utility(a, y, ap) = -inf; % Set utility to -inf if consumption is non-positive
    else
        if gamma == 1
            utility(a, y, ap) = log(c); % Utility for log preferences
    else
            utility(a, y, ap) = (c^(1 - gamma)) / (1 - gamma); % Utility for CRRA preferencesendend
        end
    end
end
end
end

%%% VFI Preallocations and Tolerances
tol=10^(-9);            %   Maximum error tolerance       
maxits=10^4;            %   Maximum number of iterations

% Initialize the value function with zeros
V0 = zeros(a_n, Y_n); % Initial guess of value function 
V1 = V0; % Preallocation of the updated value function
a_prime_index = zeros(a_n, Y_n); % Preallocation of the asset choice policy function index 

%%  Main VFI Loop
count = 0;
dif = Inf; % Initialize dif with Inf to ensure the loop starts
while dif > tol && count < maxits
for y = 1:Y_n
for a = 1:a_n
% Compute the value for all possible asset choices
V_candidate = squeeze(utility(a, y, :)) + beta * V0 * P(y, :)';
% Find the maximum value and the corresponding asset index
[V1(a, y), a_prime_index(a, y)] = max(V_candidate); 
end
end
dif = max(max(abs(V1 - V0))); % Update the maximum difference
V0 = V1; % Update the value function for the next iteration 
count = count + 1; % Increment the iteration count
end
%%% Recovery of Consumption Policy Function
a_prime = A(a_prime_index); % Asset policy function using the index 
c_policy = (1 + r) * repmat(A, 1, Y_n) + exp(repmat(Y, a_n, 1)) - a_prime;
%%  Plots

figure(1)
plot(A,V1(:,1),A,V1(:,Y_n/2+0.5),A,V1(:,Y_n))
xlabel('Assets')
ylabel('Value')
title('Value Function')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

figure(2)
plot(A,c_policy(:,1),A,c_policy(:,Y_n/2+0.5),A,c_policy(:,Y_n))
xlabel('Assets')
ylabel('Consumption')
title('Consumption')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

figure(3)
plot(A,a_prime(:,1),A,a_prime(:,Y_n/2+0.5),A,a_prime(:,Y_n))
xlabel('Assets')
ylabel('Assets')
title('Optimal Savings')
legend('Minimum Income','Steady State Income','High Income','location','southoutside','orientation','horizontal')

%%  Simulations

sims=1000;
y_sim=simulate(dtmc(P),sims-1);
a_index=1;

for t=1:sims
    c_sim(t)=(1+r)*A(t)+exp(Y(y_sim(t)))-a_prime(1,y_sim(t));
    a_index(t+1)=a_prime(t,y_sim(t));
    a_sim(t+1)=a_prime(t,y_sim(t));
end

figure(4)
subplot(3,1,1)
plot(sims/2+1:sims,exp(Y(y_sim(sims/2+1:sims))),sims/2+1:sims,ones(sims/2,1))
xlabel('Time')
ylabel('Income')

subplot(3,1,2)
plot(sims/2+1:sims,c_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Consumption')

subplot(3,1,3)
plot(sims/2+1:sims,a_sim(sims/2+1:sims))
xlabel('Time')
ylabel('Assets')

%%  Correlation

[clgm,lags] = xcorr(y_sim,c_sim,4);
figure;
plot(lags,clgm);
xlabel("lags")
ylabel("Correlation Vector")
title("Correlogram between Simulated Income and Consumption Series")
legend("Correlation at 4 lags")
