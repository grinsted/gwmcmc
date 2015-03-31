function [C,lags]=eacorr(m)
%% EACORR, Ensemble auto correlation 
% 
% Ensemble average auto correlation. 
% 
% USAGE: [C,lags]=eacorr(m)
%
% eacorr is designed to be applied on the MxWxT matrices output from gwmcmc,
% but can also be applied to 2d and 1d matrices. 
%
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
    C(:,mix)=c./(T-lags); %biased/unbiased 
    C(:,mix)=C(:,mix)./C(1,mix);
end
if isreal(m)
    C=real(C);
end

if nargout==0
    plot(lags,C,'.-',lags([1 end]),[0 0],'k');
    grid on
    xlabel('lags')
    ylabel('autocorrelation');
    clearvars lags C
end
