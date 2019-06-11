%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** Ozellikler.m dosyasi***********************************%
%***********************************************************%

fileID = fopen('test1.txt','w');

%342 Ornek

for Nx=1:342
    s=['s (',num2str(Nx),').wav'];
    [y ,fs]=audioread(s);
    y=y(:,1);
    
    x=y( floor( length(y) / 4 ) : floor( 3*length(y) / 4 ) );
    dt = 1/fs;
    t = 0:dt:length(x)*dt-dt;
    
    [b0,a0] = butter(2,325/(fs/2));
    xin = abs(x);
    xin = filter(b0,a0,xin);
    xin = xin-mean(xin);
    x2 = zeros(length(xin),1);
    x2(1:length(x)-1) = xin(2:length(x));
    zc = length(find((xin>0 & x2<0) | (xin<0 & x2>0)));
    
    F0 = 0.5*fs*zc/length(x);
    %fprintf('Zero-crossing F0 estimate is %3.3f Hz.\n',F0)
    
    data=x';
    data= filter(b0,a0,data);
    NN=length(data);
    %sound(data,fs)
    ms=1/F0;
    frame=floor(ms*fs);
    
    duration=length(data)/fs;
    peaks=zeros(1,length(data));
    for i=2:length(data)-1
        if( data(i) > data(i-1) && data(i) > data(i+1) && data(i)>0 )
            peaks(i)=1;
        end
        if( data(i) < data(i-1) && data(i) < data(i+1) && data(i)<0 )
            peaks(i)=1;
        end
    end
    peakone=find(peaks==1);
    pe=data(peakone);
    
    %stem(1:length(peaks),peaks,'.');hold on %%sadece peaklerden olusan bir dizi
    
    peaks=peaks.*data;
    %plot(1:length(data),data);hold on
    
    maxes=zeros(1,length(data));
    for i=1:frame:length(data)-frame
        
        %   z(1+floor(i/frame))=max(y(i:i+frame));
        [z,l]=max(abs(peaks(i:i+frame)));
        maxes(l+i)=1;
    end
    maxone=find(maxes==1);
    m=data(maxone);
    q=0;
    for i=2:length(m)
        if abs(m(i))<0.75*abs(m(i-1))
            maxes(maxone(i))=0;
        end
    end
    
    ppeak_counter=0;
    npeak_counter=0;
    
    [m2,maxone2]=find(maxes==1);
    
    for i=1:length(m2)
        if data(maxone2(i))>0
            ppeak_counter=ppeak_counter+1;
        elseif data(maxone2(i))<0
            npeak_counter=npeak_counter+1;
        end
    end
    
    if Nx~=142 && Nx~=86 && Nx~=113 && Nx~=24 && Nx~=245 && Nx~=274 && Nx~=304 && Nx~=305 
        if ppeak_counter > npeak_counter
            maxes( find( data< 0 ) )=0;
        else
            maxes( find( data > 0 ) )=0;
        end
    end
    
    [m2,maxone2]=find(maxes==1);
    
    %stem(1:length(maxes),maxes,'.');hold on;
    
    T=diff(maxone2);
    Tavg=mean(T);
    T(find(T<0.76*Tavg))=0;
    T(find(T>1.3*Tavg))=0;
    
    t0=find(T~=0);
    T0=T(t0);
    
    pitch_length=T0*duration/length(data);
    
    %kisi_no%
    kisi_no=ceil(Nx/6);
    fprintf(fileID,'%d',kisi_no);
    fprintf(fileID,',');
    
    %jitta%
    ji=abs(diff(pitch_length));
    jitta=mean(ji);
    %fprintf('jitta is %8.8f.\n',jitta)
    fprintf(fileID,'%8.8f',jitta);
    fprintf(fileID,',');
    
    %jitt%
    payda=mean(pitch_length);
    jitt=(jitta/payda)*100;
    %fprintf('jitt is %8.8f.\n',jitt)
    fprintf(fileID,'%8.8f',jitt);
    fprintf(fileID,',');
    
    %jit_rap %
    rap3=zeros( 1 , length(pitch_length)-2 );
    for i=2 : (length(pitch_length)-1)
        rap3(i-1)=sum( pitch_length( (i-1) : (i+1) ) )/3;
    end
    jit_rap =100 * mean( abs( pitch_length(2 : (end-1) ) - rap3 ) ) / payda;
    %fprintf('jit_rap is %8.8f.\n',jit_rap)
    fprintf(fileID,'%8.8f',jit_rap);
    fprintf(fileID,',');
    
    %jit_ppq5%
    ppq5=zeros( 1 , length(pitch_length)-4 );
    for i=3 : (length(pitch_length)-2)
        ppq5(i-2)=sum( pitch_length( (i-2) : (i+2) ) )/5;
    end
    jit_ppq5 =100 * mean( abs( pitch_length(3 : (end-2) ) - ppq5 ) ) / payda;
    %fprintf('jit_ppq5 is %8.8f.\n',jit_ppq5)
    fprintf(fileID,'%8.8f',jit_ppq5);
    fprintf(fileID,',');
    
    %jitt_DDP%
    ddp=zeros(1,length(pitch_length)-2);
    for i=2:length(pitch_length)-1
        ddp(i-1)=abs( ( pitch_length(i+1)-pitch_length(i) ) - ( pitch_length(i)-pitch_length(i-1) ) );
    end
    jitt_DDP=mean(ddp);
    %fprintf('jitt_DDP is %8.8f.\n',jitt_DDP)
    fprintf(fileID,'%8.8f',jitt_DDP);
    fprintf(fileID,',');
    
    %shimer parameteres%
    A=abs(data(maxone2));
    Aavg=mean(A);
    
    %shDB%
    shimdb=zeros(1,length(maxone2)-1);
    for i=1:length(A)-1
        shimdb(i)= 20 * abs ( log10( A(i+1) / A(i) ) );
    end
    shDB=mean(shimdb);
    %fprintf('shDB is %8.8f.\n',shDB)
    fprintf(fileID,'%8.8f',shDB);
    fprintf(fileID,',');
    
    %shim%
    shim= 100 * mean( abs( A(1:(end-1)) - A(2:end) ) )/Aavg;
    %fprintf('shim is %8.8f.\n',shim)
    fprintf(fileID,'%8.8f',shim);
    fprintf(fileID,',');
    
    %sh_apq3%
    apq3_pay=zeros(1,length(A)-2);
    for i=2:length(A)-1
        apq3_pay(i-1)= A(i) - mean( A( (i-1):(i+1) ) );
    end
    sh_apq3= 100*mean( abs(apq3_pay) ) / Aavg;
    %fprintf('sh_apq3 is %8.8f.\n',sh_apq3)
    fprintf(fileID,'%8.8f',sh_apq3);
    fprintf(fileID,',');
    
    %sh_apq5%
    apq5_pay=zeros(1,length(A)-4);
    for i=3:length(A)-2
        apq5_pay(i-2)= A(i) - mean( A( (i-2):(i+2) ) );
    end
    sh_apq5= 100*mean( abs(apq5_pay) ) / Aavg;
    %fprintf('sh_apq5 is %8.8f.\n',sh_apq5)
    fprintf(fileID,'%8.8f',sh_apq5);
    fprintf(fileID,',');
    
    %sh_apq11%
    apq11_pay=zeros(1,length(A)-10);
    for i=6:length(A)-5
        apq11_pay(i-5)= A(i) - mean( A( (i-5):(i+5) ) );
    end
    sh_apq11= 100*mean( abs(apq11_pay) ) / Aavg;
    %fprintf('sh_apq11 is %8.8f.\n',sh_apq11)
    fprintf(fileID,'%8.8f',sh_apq11);
    fprintf(fileID,',');
    
    %sh_DDP%
    shddp=zeros(1,length(A)-2);
    for i=2:length(A)-1
        shddp(i-1)=abs( ( A(i+1)-A(i) ) - ( A(i)-A(i-1) ) );
    end
    shim_DDP=mean(shddp);
    %fprintf('shim_DDP is %8.8f.\n',shim_DDP)
    fprintf(fileID,'%8.8f',shim_DDP);
    fprintf(fileID,',');
    
    %median_pitch%
    median_pitch=median(pitch_length);
    %fprintf('median_pitch is %8.8f.\n',median_pitch)
    fprintf(fileID,'%8.8f',median_pitch);
    fprintf(fileID,',');
    
    %mean_pitch%
    mean_pitch=mean(pitch_length);
    %fprintf('mean_pitch is %8.8f.\n',mean_pitch)
    fprintf(fileID,'%8.8f',mean_pitch);
    fprintf(fileID,',');
    
    %max_pitch%
    max_pitch=max(pitch_length);
    %fprintf('max_pitch is %8.8f.\n',max_pitch)
    fprintf(fileID,'%8.8f',max_pitch);
    fprintf(fileID,',');
    
    %min_pitch%
    min_pitch=min(pitch_length);
    %fprintf('min_pitch is %8.8f.\n',min_pitch)
    fprintf(fileID,'%8.8f',min_pitch);
    fprintf(fileID,',');
    
    %range_pitch%
    range_pitch=max_pitch-min_pitch;
    %fprintf('range_pitch is %8.8f.\n',range_pitch)
    fprintf(fileID,'%8.8f',range_pitch);
    fprintf(fileID,',');
    
    %s_sapma_pitch% %standart deviation%
    s_sapma_pitch=std(pitch_length);
    %fprintf('standart_sapma_pitch is %8.8f.\n',s_sapma_pitch)
    fprintf(fileID,'%8.8f',s_sapma_pitch);
    fprintf(fileID,',');
    
    %otoKT%
    T0=floor(fs/F0);
    otoKT=(1/(NN-T0))*sum( abs(data(T0:NN-1)).*abs(data(1:NN-T0)) );
    %fprintf('otoKT is %8.8f.\n',otoKT)
    fprintf(fileID,'%8.8f',otoKT);
    fprintf(fileID,',');
    
    %otoK%
    otoK= (1/NN)*sum( abs(data(1:NN-1)).*abs(data(1:NN-1)) );
    %fprintf('otoK is %8.8f.\n',otoK)
    fprintf(fileID,'%8.8f',otoK);
    fprintf(fileID,',');
    
    %HNR %
    HNR=otoKT/(otoK-otoKT);
    %fprintf('HNR is %8.8f.\n',HNR)
    fprintf(fileID,'%8.8f',HNR);
    fprintf(fileID,',');
    
    %NHR%
    NHR=1/HNR;
    %fprintf('NHR is %8.8f.\n',NHR)
    fprintf(fileID,'%8.8f',NHR);
    fprintf(fileID,',');
    
    %class%
    if (Nx<=174)
        hasta=0;
    else
        hasta=1;
    end
    fprintf(fileID,'%d',hasta);
    
    %new line%
    fprintf(fileID,'\n');
    
end

fclose(fileID);




