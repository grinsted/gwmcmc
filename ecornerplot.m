function H=ecornerplot(m,varargin)
%% Corner plot with allowance for effective sample size
%
% ecornerplot(m,[parameter,value])
% 
% INPUTS:
%   m: a matrix of values that should be plotted in the corner plot. 
%
% 
%   When m is a 3d matrix (ndims(m)==3), then it is assumed to have the form 
%   MxWxT as output from GWMCMC, where M is the number of parameters, W is
%   the number of walkers, and T is the number of steps in each markov chain.
%
%
% NAMED PARAMETERS:
%   range: Restrict visual limits to central [99.5] percentile.
%   names: A cell of strings with labels for each parameter in m.
%   ks: enable kernel smoothing density instead of histograms [false]
%   support: a 2xM matrix with upper and lower limits.
%   ess: effective sample size. [default: auto-determine ess using EACORR.]
%        - used to adjust bandwidth estimates in kernel density estimates.
%   scatter: show scatter plot instead of 2d-kernel density estimate [true if #points<2000]. 
%   fullmatrix: display upper corner of plotmatrix. [false]
%   color: A color-theme for the plot. [.5 .5 .5].
%   grid: show grid. [false].
%
%
%  Notes: The 2d kernel density contours are plotted as the highest density
%  contours at intervals of 10%:20%:90% 
%
% EXAMPLE: 
%   mu = [1 -1 1]; C= [.9 .4 .1; .4 .3 .2; .1 .2 1];
%   m=mvnrnd(mu,C,6000);
%   m(:,3)=exp(m(:,3)/2);
%   ecornerplot(m,'support',[nan nan;nan nan; 0 nan]','ks',true);
%
% Aslak Grinsted 2015

if nargin==0
    close all
    m=randn(10000,4);%m(:,2)=m(:,2)+100000;
    ecornerplot(m,'ks',true,'fullmatrix',false,'grid',true);
    error('test mode... ')
    return
end

p = inputParser;
p.addOptional('range',99.5,@isnumeric);
p.addOptional('names',{},@iscellstr);
p.addOptional('ks',false,@islogical);  %TODO: allow definition of support?
p.addOptional('support',[]);
p.addOptional('grid',false,@islogical);
p.addOptional('scatter',nan,@islogical);
p.addOptional('fullmatrix',false,@islogical);
p.addOptional('color',[1 1 1]*.5,@(x)all(size(x)==[1 3]))
p.addOptional('ess',[]);
% p.addOptional('truth',[fa],@isnumeric);
p.parse(varargin{:});
p=p.Results;


if (size(m,1)<size(m,2))&&(ismatrix(m)), m=m'; end; %Consider this behaviour further....


if isempty(p.ess)
    [~,~,p.ess]=eacorr(m);
    p.ess=mean(p.ess);
end



if ndims(m)==3
    m=m(:,:)'; 
end




M=size(m,2);
Np=size(m,1);
if p.ess>Np
    error('Effective Sample Size (ess) must be smaller than number of samples')
end
if M>20
    error('Too many dimensions. You probably don''t want to make that many subplots. ')
end
if isnan(p.scatter)
    p.scatter=Np<2000;
end

p.range=prctile(m,[50+[-1 1]*p.range/2 0 100]); %first 2 
rng=p.range(4,:)-p.range(3,:);
if isempty(p.support),p.support=nan(2,M);end
ix=isnan(p.support(1,:)); p.support(1,ix)=p.range(3,ix)-rng(ix)/4;
ix=isnan(p.support(2,:)); p.support(2,ix)=p.range(4,ix)+rng(ix)/4;




for ii=length(p.names)+1:M
    p.names{ii}=sprintf('m_{%.0f}',ii);
end
% for ii=size(p.truth,2)+1:M
%     p.truth(ii,:)=nan;
% end
if p.grid
    p.grid='on';
else
    p.grid='off';
end

clf
H=nan(M);
for r=1:M
    for c=1:max(r,M*p.fullmatrix)
        H(r,c)=subaxis(M,M,c,r,'s',0.01,'mb',0.12,'mt',0.05,'ml',0.12,'mr',0.0);
        if c==r
            if p.ks
                [F,X,bw]=ksdensity(m(:,r),'support',p.support(:,r)); %TODO: use ESS 
                if p.ess<Np
                    [F,X,bw]=ksdensity(m(:,r),'width',bw*(Np/p.ess)^.2,'support',p.support(:,r)); %(the power 1/5 comes from examining the bandwidth calculation in ksdensity)
                end
                X=X([1,1:end,end]);F=[0,F,0];
            else
                [F,X]=histcounts(m(:,r),'Normalization','pdf');
                X=X(ceil(0.5:0.5:end));
                F=[0,F(ceil(0.5:0.5:end)),0];
            end
            fill(X,F,p.color,'edgecolor','none')
            set(gca,'ytick',[],'YLim',[0 max(F)*1.1])
            set(gca,'XGrid',p.grid)
        else
            if p.scatter
                plot(m(:,c),m(:,r),'.','color',p.color)
            else
                %                 [N,C]=hist3(m(:,[c r]),[0 0]+ceil(sqrt(Np)/5));
                %                 imagesc(C{1},C{2},N)
                %                 caxis([0 max(N(:))]);
                %                 axis xy
                try
                    [~,N,X,Y]=kde2d(m(:,[c r]),2^9,p.support(1,[c r]),p.support(2,[c r]),p.ess);
                    %                 ns=sort(N(:));
                    %                 cint=interp1q(cumsum(ns)/sum(ns),ns,[0.05 0.17 0.50 0.83 0.95]');
                    hold on
                    %N=N/max(N(:));
                    %contourf(X,Y,N,(0.1:.2:1)','edgecolor',p.color); %TODO: try to make it HDI like???
                    N=N/sum(N(:));
                    NS=sort(N(:));
                    levels = interp1q(cumsum(NS),NS,(0.1:.2:1)')'; %HDI LEVELS
                    contourf(X,Y,N,levels,'edgecolor',p.color);
                    caxis([0,max(NS)])
                catch
                end
                %                 pcolor(X,Y,N); %
                %                 shading interp
            end
            set(gca,'XGrid',p.grid,'YGrid',p.grid)
            if diff(p.range(1:2,r))>0, set(gca,'Ylim',p.range(1:2,r)); end
        end
        if r==M, xlabel(['^{ }' p.names{c} '_{ }']);end
        if (c==1)&(r>1-p.fullmatrix), ylabel(['^{ }' p.names{r} '_{ }']);end
        if diff(p.range(1:2,c))>0, set(gca,'Xlim',p.range(1:2,c)'); end
    end
    
end
h=H(:,2:end);h(isnan(h))=[];
set(h,'YTickLabel',[])
h=H(1:M-1,:);h(isnan(h))=[];
set(h,'XTickLabel',[])
colormap(bsxfun(@minus,[1 1 1],linspace(0,.7,300)'));


%LINK the axes for zooming:
hlink={};
drawnow
lh=cellfun(@(x)double(x),get(H(~isnan(H)),'Xlabel'));
set(lh,'units','normalized')
set(lh,'position',min(cell2mat(get(lh,'position'))));
hlink{end+1}=linkprop(lh,'position');
lh=cellfun(@(x)double(x),get(H(~isnan(H)),'Ylabel'));
set(lh,'units','normalized');
set(lh,'position',min(cell2mat(get(lh,'position'))));
hlink{end+1}=linkprop(lh,'position');

for ii=1:M
    h=H(:,ii); h(isnan(h))=[];
    set(h,'XLimMode','manual')
    hlink{end+1}=linkprop(h,'XLim');
    h=H(ii,1:ii-1); h(isnan(h))=[];
    set(h,'YLimMode','manual')
    hlink{end+1}=linkprop(h,{'YLim','YTick'});
end
setappdata(gcf,'aplotmatrix_linkprop_handles',hlink)


if nargout==0
    clearvars H
end
