function [C,lags,ESS]=eacorr(m)
%% EACORR, Ensemble auto correlation 
% 
% Ensemble average auto correlation. 
%
% Useful if you want to estimate the ACF using multiple MCMC chains at once. 
% 
% USAGE: [C,lags,ESS]=eacorr(m)
%
% INPUTS: 
%    m: eacorr is designed to be applied on the MxWxT matrices output from gwmcmc,
%       but can also be applied to 2d and 1d matrices. 
% 
% OUTPUTs
%    C: Ensemble average autocorrelation function estimated for each model
%    dimension independently
%    lags: lags corresponding to each row in C.
%    ESS: Effective Sample Size estimated from the autocorrelation function of each model dimension. 
%
% Details:
% When m is a 3d matrix of size MxWxT: 
%  * the auto-correlation will be calculated along the T-dimension
%  * the ensemble averaging will be done along the W-dimension
%  * the auto-correlation will be calculated for each of the M-dimensions
%
% When m is a 1d or 2d matrix, then it will be assumed that there is just a
% single ensemble member, and the autocorrelation will be measured along
% the longest dimension.
%
% (The centering of the input series is done using the ensemble mean rather 
% than the mean of each individual series.)
%
% example:
%    eacorr(filter2(ones(100,1),randn(1000,10)))
%
% Aslak Grinsted 2015


sM=size(m);
sM(end+1:3)=1;

if sM(3)==1
    if sM(1)>sM(2)
        m=permute(m,[2 3 1]);
    else
        m=permute(m,[1 3 2]);
    end
end


M=size(m,1);W=size(m,2);T=size(m,3);
N=W*T;

lags=(0:T-1)';
nfft = 2^nextpow2(2*T-1);
C=nan(T,M);
for mix=1:M
    c=zeros(T,1);
    mm=mean(m(mix,:)); %center data using ensemble average!
    for wix=1:W
        d=m(mix,wix,:);
        d=d(:)-mm;
        r=ifft( abs(fft(d,nfft)).^2);
        c=c+r(1:T);
    end
%     C(:,mix)=c./(T-lags); %biased/unbiased 
%     C(:,mix)=C(:,mix)./C(1,mix);
    C(:,mix)=c./c(1);
end
if isreal(m)
    C=real(C);
end

if nargout>2
    ESS=nan(1,size(C,2));
    %we use N/(1+2*sum(ACF)) (eqn 7.11 in DBDA2)
    %Here, I assume that the ACF can be approximated with exp(-k/lag)
    
    for ii=1:size(C,2)
        kix=find(C(:,ii)<=0.5,1);%we determine k at the lag where C~=0.5;
        if isempty(kix), kix=2; end %use lag1 as fall-back for short chains. TODO:warn? 
        if (C(kix,ii)<0.05)&&(kix==2), ESS(ii)=N;continue;end %essentially no autocorrelation...
        k=-log(C(kix,ii))./lags(kix);
        sumACF=1/(exp(k)-1); %http://functions.wolfram.com/ElementaryFunctions/Exp/23/01/0001/
        ESS(ii)=N/(1+2*sumACF);
    end
end

if nargout==0
    plot(lags,C,'.-',lags([1 end]),[0 0],'k');
    grid on
    xlabel('lags')
    ylabel('autocorrelation');
    clearvars lags C ESS
end
