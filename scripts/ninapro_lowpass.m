function ninapro_lowpass(in,out)
  for subject=0:26
    for gesture=1:52
      for trial=0:9
        path = sprintf('%03d/%03d/%03d_%03d_%03d.mat',...
          subject,gesture,subject,gesture,trial);
        fprintf([path '\n']);
        deal_one(in,out,path);
      end
    end
  end
end

function deal_one(in,out,path)
  in = [in '/' path];
  out = [out '/' path];
  dir = out(1:end-16);
  if ~exist(dir,'dir')
    mkdir(dir);
  end
  f = load(in);
  data = f.data;
  label = f.label;
  subject = f.subject;
  parfor ch=1:10
    data(:,ch) = lowpass(data(:,ch));
  end
  save(out,'data','label','subject');
end

function y = lowpass(x)
  fc = 1;
  fs = 100;
  [b,a] = butter(1,fc/(fs/2));
  y = filtfilt(b,a,x);
end