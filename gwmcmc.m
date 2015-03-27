function [models,logP]=gwmcmc(minit,logPfuns,mccount,stepsize,skip)
%% Cascaded affine invariant ensemble MCMC sampler. "The MCMC hammer"
% 
% GWMCMC is an implementation of the Goodman and Weare 2010 Affine
% invariant ensemble Markov Chain Monte Carlo (MCMC) sampler. MCMC sampling
% enables bayesian inference. The problem with many traditional MCMC samplers
% is that they can have slow convergence for badly scaled problems, and that
% it is difficult to optimize the random walk for high-dimensional problems.
% This is where the GW-algorithm really excels as it is affine invariant. It
% can achieve much better convergence on badly scaled problems. It is much
% simpler to get to work straight out of the box, and for that reason it
% truly deserves to be called the MCMC hammer.  
% 
% (This code uses a cascaded variant of the Goodman and Weare algorithm).  
%
% USAGE:
%  [models,logP]=gwmcmc(minit,logPfuns,mccount,stepsize,skip);
%
% INPUTS:
%     minit: an MxW matrix of initial values for each of the walkers in the
%            ensemble. (M:number of model params. W: number of walkers). W
%            should be atleast 2xM. (see e.g. mvnrnd). 
%  logPfuns: a cell of function handles returning the log probality of a
%            proposed set of model parameters. Typically this cell will
%            contain two function handles: one to the logprior and another 
%            to the loglikelihood. E.g. {@(m)logprior(m) @(m)loglike(m)}
%   mccount: What is the desired total number of monte carlo proposals.
%            This is the total number, -NOT the number per chain.
%  stepsize: unit-less stepsize (default=2.5). 
%      skip: Thin all the chains by only storing every N'th step [default=10]
%
% OUTPUTS:
%    models: A MxWxT matrix with the thinned markov chains (with T samples
%            per walker). T=~mccount/skip/W.
%      logP: A PxWxT matrix of log probabilities for each model in the
%            models. here P is the number of functions in logPfuns. 
%
% Note on cascaded evaluation of log probabilities: 
% The logPfuns-argument can be specifed as a cell-array to allow a cascaded 
% evaluation of the probabilities. The computationally cheapest function should be
% placed first in the cell (this will typically the prior). This allows the
% routine to avoid calculating the likelihood, if the proposed model can be
% rejected based on the prior alone. 
% logPfuns={logprior loglike} is faster but equivalent to
% logPfuns={@(m)logprior(m)+loglike(m)}
%
% TIP: if you aim to analyze the entire set of ensemble members as a single
% sample from the distribution then you may collapse output models-matrix
% thus: models=models(:,:); This will reshape the MxWxT matrix into a 
% Mx(W*T)-matrix while preserving the order.
%
%
% EXAMPLE:
% Here we sample a multivariate normal distribution.
% %define problem:
% mu = [5;-3;6];
% C = [.5 -.4 0;-.4 .5 0; 0 0 1];
% iC=pinv(C);
% logPfuns={@(m)-0.5*sum((m-mu)'*iC*(m-mu))}
%
% %make a set of starting points for the entire ensemble of walkers
% minit=randn(length(mu),length(mu)*2);
%
% %Apply the MCMC hammer
% [models,logP]=gwmcmc(minit,logPfuns,100000);
% models(:,:,1:floor(end/5))=[]; %remove 20% as burn-in 
% models=models(:,:)'; %reshape matrix to collapse the ensemble member dimension
% scatter(models(:,1),models(:,2))
% prctile(models,[5 50 95])
%
%
% References:
% Goodman & Weare (2010), Ensemble Samplers With Affine Invariance, Comm. App. Math. Comp. Sci., Vol. 5, No. 1, 65–80
% Foreman-Mackey, Hogg, Lang, Goodman (2013), emcee: The MCMC Hammer, arXiv:1202.3665
% 
% -Aslak Grinsted 2015


% % test-code
% if nargin==0
% 
%     close all
%        
%     
%     %--- Setup the problem --- 
%     mu = [5];
%     logPfun=@(m)-0.5*sum((m-mu).^2/1);
%     
%     %logPfun=@(m) (-100*(m(2)-m(1).^2).^2 -(1-m(1)).^2)/20; %the rosenbrock banana.
%     
%     %Setup the starting point of all walkers in the ensemble.
%     minit=randn(1,10)+5;
%     
%     [models,logP]=gwmcmc(minit,logPfun,100000,9); %Apply the MCMC hammer
%     
% %    models(:,:,1:1000)=[];%crop 20% burn-in
%     
%     models=models(:,:);
%     std(models(:))
% 
% %     scatter(models(1,:),models(2,:),[],1:length(models(2,:)),'.');
% %     colormap(cool)
%     
%     clearvars models logP
%     
%     return
% end

if nargin<3
    error('GWMCMC requires atleast 3 inputs.')
end

M=size(minit,1);

if size(minit,2)==1 
    minit=bsxfun(@plus,minit,randn(M,M*5)); 
end


Nwalkers=size(minit,2);

if (nargin<5)||isempty(skip)
    skip=10;
end
if (nargin<4)||isempty(stepsize)
    stepsize=2.5;
end
if size(minit,1)*2>size(minit,2)
    warning('Check minit dimensions.\nIt is recommended that there be atleast twice as many walkers in the ensemble as there are model dimension. ')
end



Nkeep=ceil(mccount/skip/Nwalkers); %number of samples drawn from each walker
mccount=(Nkeep-1)*skip+1;

models=nan(M,Nwalkers,Nkeep); %pre-allocate output matrix

models(:,:,1)=minit;

if ~iscell(logPfuns)
    logPfuns={logPfuns};
end

NPfun=numel(logPfuns);

%calculate logP state initial pos of walkers
logP=nan(NPfun,Nwalkers,Nkeep);
for wix=1:Nwalkers
    for fix=1:NPfun
        logP(fix,wix,1)=logPfuns{fix}(minit(:,wix));
    end
end
if ~all(all(isfinite(logP(:,:,1))))
    error('Starting points for all walkers must have finite logP')
end

%reject=zeros(1,NPfun+1); %keep track of how many is being rejected by each logPfun.
%accept=zeros(1,NPfun+1);
reject=zeros(Nwalkers,1);

% hwait = waitbar(0, 'Ensemble Markov Chain Monte Carlo','name','MCMC');
% pos=get(hwait,'pos');
% set(hwait,'pos',pos+[0 0 0 min(M,10)*12])


ctime=cputime;
starttime=cputime;

curm=models(:,:,1); 
curlogP=logP(:,:,1);
progress(0)
for ii=2:mccount
    %generate proposals for all walkers (done outside loop, to be compatible with parfor):
    rix=mod((1:Nwalkers)+floor(rand*(Nwalkers-1)),Nwalkers)+1; %pick a random partner 
    zz=((stepsize - 1)*rand(1,Nwalkers) + 1).^2/stepsize;
    proposedm=curm(:,rix) - bsxfun(@times,(curm(:,rix)-curm),zz);
    
    acceptfullstep=true(1,Nwalkers);

    logrand=log(rand(NPfun+1,Nwalkers)); %moved outside because rand is slow inside parfor
    for wix=1:Nwalkers %TODO:PARFOR
        lr=logrand(:,wix);
        cp=curlogP(:,wix);
        proposedlogP=nan(NPfun,1);
        if lr(1)<(M-1)*log(zz(wix))
            for fix=1:NPfun
                proposedlogP(fix)=logPfuns{fix}(proposedm(:,wix));
                if lr(fix+1)>proposedlogP(fix)-cp(fix)
                    reject(wix)=reject(wix)+1;
                    acceptfullstep(wix)=false;
                    break
                end
            end
        else
            reject(wix)=reject(wix)+1;
            acceptfullstep(wix)=false;
        end
        if acceptfullstep(wix)
            curm(:,wix)=proposedm(:,wix);
            curlogP(:,wix)=proposedlogP;
        end
    end
    if mod(ii-1,skip)==0
        row=ceil(ii/skip);
        models(:,:,row)=curm;
        logP(:,:,row)=curlogP;
    end
    
    %progress bar
    progress(ii/mccount,curm,sum(reject)/(ii*Nwalkers))

end
progress(1);


% TODO: make standard diagnostics to give warnings...
% TODO: cut away initial drift.(?)
% TODO: make some diagnostic plots if nargout==0;




function progress(pct,curm,rejectpct)
    persistent lastNchar lasttime starttime
    if isempty(lastNchar)||pct==0
        lastNchar=0;lasttime=cputime-10;starttime=cputime;fprintf('\n')
        pct=1e-16;
    end
    if pct==1
        fprintf('%s',repmat(char(8),1,lastNchar));lastNchar=0;
        return
    end
    if (cputime-lasttime>0.1)
        
        ETA=datestr((cputime-starttime)*(1-pct)/(pct*60*60*24),13);
        progressmsg=[uint8((1:40)<=(pct*40)).*'#' ''];
        curmtxt=sprintf('% 9.3g\n',curm(1:min(end,20),1));
        %curmtxt=mat2str(curm);
        progressmsg=sprintf('GWMCMC %5.1f%% [%s] %s\n%3.0f%% rejected\n%s\n',pct*100,progressmsg,ETA,rejectpct*100,curmtxt);
        
        fprintf('%s%s',repmat(char(8),1,lastNchar),progressmsg);
        drawnow;lasttime=cputime;
        lastNchar=length(progressmsg);
    end
    
    
% Acknowledgements: I became aware of the algorithm via a student report 
% which was using emcee for python. I read the paper and judged that this
% must be excellent, and made my own implementation for matlab. It is
% excellent. 