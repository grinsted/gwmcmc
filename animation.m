n=netcdfobj('C:\Users\Aslak\HugeData\GriddedData\had4_krig_v2_0_0.nc');

T=permute(n.vars.temperature_anomaly.value,[2 1 3]);
x=n.vars.longitude.value;
y=n.vars.latitude.value;
[X,Y]=meshgrid(x,y);
up=5;
X=imresize(X,up,'bilinear');Y=imresize(Y,up,'bilinear');
t=double(n.vars.time.value);
ix=t<datenum(1920,1,1);
mT=mean(T(:,:,ix),3);
T=bminus(T,mT);

w=12*5;
for c=1:length(x)
    for r=1:length(y)
        TT=T(r,c,:);
        T(r,c,:)=smooth(TT(:),w);
    end
end

T(:,:,1:floor(w/2))=[];
T(:,:,end-floor(w/2):end)=[];
t=t(floor(w/2)+(1:size(T,3)));



vidObj = VideoWriter('CowtanWay.mp4','MPEG-4');
vidObj.FrameRate=25;
vidObj.Quality=95;
open(vidObj);

    
close all
set(gcf,'position',[100 100 1280 768])
hax1=axes('position',[0 0 .9 1]);
hold on
bluemarble
meanT=mean(T(:));sigmaT=std(T(:));
caxis([-1 1]*prctile(T(:),99.9));
clim=caxis;
haxbar=axes('position',[.92 .1 .05 .8]);
Yb=linspace(prctile(T(:),.01),clim(2),300)';
imagesc(0,Yb,Yb);
caxis(clim)
axis xy off
ylabels=-3:3;
for ii=1:length(ylabels)
    ltext(0,ylabels(ii),sprintf('%.0f^oC',ylabels(ii)),'mc','clip','on','color',[0 0 0]);
end
wt=cos(Y*pi/180);wt=wt/sum(wt(:));
hold on
hline=plot([-.5 .5],[0 0],'k','linewidth',3);




axes(hax1)
hslcolormap('mbbbc.YRRRm',1,[.2 .95 .1]);
axis off
hh=[];
hold on
htxt=ltext(0.5,0.05,'Temperature above 1850-1920','nbc','fontsize',18,'color',[0 0 0]);
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);    

for ii=1:3:length(t)
    S=imresize(T(:,:,ii),up,'lanczos2');
    smT=sum(S(:).*wt(:));
    set(hline,'ydata',[0 0]+smT);
    axes(hax1)
    set(htxt,'string',datestr(t(ii),'yyyy'))
    delete(hh)
    hh=alphawarp(X,Y,S,1-exp(-0.5*((S-meanT)/sigmaT).^2));
    uistack(htxt,'top')

    drawnow
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);    
end
close(vidObj);
clear vidObj

