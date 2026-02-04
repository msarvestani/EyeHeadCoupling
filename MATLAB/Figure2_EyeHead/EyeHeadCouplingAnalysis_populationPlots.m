%function EyeHeadCouplingAnalysis_populationPlots()

close all

path = fullfile(cd,'analysis_final_testingplotting');
load(fullfile(path, "saccade_imu_population_table.mat"),'saccade_imu_population_table')

parameters.framerate = 61; %EK changed 7/1/25, this is roughly the average across all the sessions
parameters.hm_thresh = 2;
parameters.hm_window_frames = round(parameters.framerate/2):round(parameters.framerate)-1; %~500ms before
ts_ms = (-round(parameters.framerate):round(parameters.framerate))/parameters.framerate*1000;

plotcolors = [0.6666    0.3569    0.6314
            0.1333    0.4709    0.7098
            0.9569    0.4745    0.2353];

for i=1:height(saccade_imu_population_table)
    processed_imu = cell2mat(saccade_imu_population_table.imu_envelope(i));
    hd = zeros(size(processed_imu));
    hd(processed_imu >=parameters.hm_thresh) = processed_imu(processed_imu >=parameters.hm_thresh);

    saccade_imu_population_table.hd(i) = {sum(hd,1)>0}; % head movement yes/no at each time point
    if ~isempty(min([find(hd(1,:),1,'first') find(hd(2,:),1,'first') find(hd(3,:),1,'first')]))
        saccade_imu_population_table.onset_frame(i) = min([find(hd(1,:),1,'first') find(hd(2,:),1,'first') find(hd(3,:),1,'first')]);
        saccade_imu_population_table.onset_ms(i) = ts_ms(saccade_imu_population_table.onset_frame(i));
    else
        saccade_imu_population_table.onset_frame(i) = NaN;
        saccade_imu_population_table.onset_ms(i) = NaN;
    end

    if saccade_imu_population_table.onset_frame(i) < min(parameters.hm_window_frames) 
        saccade_imu_population_table.hm_category(i)=-1; %head movement started before window
    elseif saccade_imu_population_table.onset_frame(i) >= min(parameters.hm_window_frames) && saccade_imu_population_table.onset_frame(i) <= max(parameters.hm_window_frames)
        saccade_imu_population_table.hm_category(i)=1; %head movement started inside window
    else
        saccade_imu_population_table.hm_category(i)=0; %no head movement before or during window
    end

    saccade_imu_population_table.max_std_anywhere(i) = max(processed_imu,[],'all');
    saccade_imu_population_table.max_std_window(i) = max(processed_imu(:,parameters.hm_window_frames),[],'all');
end




filenames = unique(saccade_imu_population_table.filename);
animal_ids = unique(saccade_imu_population_table.animal_id);

% add columns for file_id and animal_num to the end of table
file_count = 1;
animal_count = 1;
for i = 1:height(saccade_imu_population_table)
    saccade_imu_population_table.file_id(i) = file_count;
    saccade_imu_population_table.animal_num(i) = animal_count;
    if i == height(saccade_imu_population_table)
        continue;
    end
    if ~strcmp(saccade_imu_population_table.filename(i),saccade_imu_population_table.filename(i+1))
        file_count=file_count+1;
    end
    if ~strcmp(saccade_imu_population_table.animal_id(i),saccade_imu_population_table.animal_id(i+1))
        animal_count=animal_count+1;
    end
end


% file summary stats
file_summary_stats = table('Size',[file_count 0]);
for i = 1:file_count
    this_file = saccade_imu_population_table(saccade_imu_population_table.file_id == i,:);
    file_summary_stats.filename(i) = this_file.filename(1);
    file_summary_stats.file_id(i) = this_file.file_id(1);
    file_summary_stats.animal_num(i) = this_file.animal_num(1);
    file_summary_stats.num_saccades(i) = height(this_file);
    file_summary_stats.session_duration(i) = this_file.session_duration(1);
    file_summary_stats.saccades_per_min(i) = height(this_file)/this_file.session_duration(1);
    file_summary_stats.avg_secs_btwn_saccades(i) = this_file.session_duration(1)*60/height(this_file);
    file_summary_stats.sacs_with_hm_ongoing(i) = sum(this_file.hm_category==-1);
    file_summary_stats.sacs_with_hm_in_window(i) = sum(this_file.hm_category==1);
    file_summary_stats.sacs_with_no_hm(i) = sum(this_file.hm_category==0);
    file_summary_stats.percent_with_ongoing_hm(i) = sum(this_file.hm_category==-1)/height(this_file);
    file_summary_stats.percent_with_hm_in_window(i) = sum(this_file.hm_category==1)/height(this_file);
    file_summary_stats.percent_with_no_hm(i) = sum(this_file.hm_category==0)/height(this_file);
end

% animal summary stats
animal_summary_stats = table('Size',[animal_count 0]);
for i = 1:animal_count
    this_file = saccade_imu_population_table(saccade_imu_population_table.animal_num == i,:);
    animal_summary_stats.file_count(i) = length(unique(this_file.filename));
    animal_summary_stats.animal_num(i) = this_file.animal_num(1);
    animal_summary_stats.animal_id(i) = this_file.animal_id(1);
    animal_summary_stats.num_saccades(i) = height(this_file);
    animal_summary_stats.session_duration(i) = sum(unique(this_file.session_duration));
    animal_summary_stats.saccades_per_min(i) = height(this_file)/animal_summary_stats.session_duration(i);
    animal_summary_stats.avg_secs_btwn_saccades(i) = animal_summary_stats.session_duration(i)*60/height(this_file);
    animal_summary_stats.sacs_with_hm_ongoing(i) = sum(this_file.hm_category==-1);
    animal_summary_stats.sacs_with_hm_in_window(i) = sum(this_file.hm_category==1);
    animal_summary_stats.sacs_with_no_hm(i) = sum(this_file.hm_category==0);
    animal_summary_stats.percent_with_ongoing_hm(i) = sum(this_file.hm_category==-1)/height(this_file);
    animal_summary_stats.percent_with_hm_in_window(i) = sum(this_file.hm_category==1)/height(this_file);
    animal_summary_stats.percent_with_no_hm(i) = sum(this_file.hm_category==0)/height(this_file);
end

saccades_with_head_movements_ongoing = saccade_imu_population_table(saccade_imu_population_table.hm_category==-1,:);
saccades_with_head_movements_in_window = saccade_imu_population_table(saccade_imu_population_table.hm_category==1,:);
saccades_without_head_movements = saccade_imu_population_table(saccade_imu_population_table.hm_category==0,:);

%% 3 eye head coupling categories
figure('position',[100         100        1500         800]); 
sac_w_hm_ongoing = height(saccades_with_head_movements_ongoing);
sac_w_hm_in_window = height(saccades_with_head_movements_in_window);
sac_wo_hm = height(saccades_without_head_movements);
percents = round([sac_w_hm_ongoing; sac_w_hm_in_window; sac_wo_hm]./height(saccade_imu_population_table)*100);
b = bar(["Head movement starts before window"; "Head movement starts within window"; "No head movement within window"],[sac_w_hm_ongoing; sac_w_hm_in_window; sac_wo_hm]);
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = strcat(string(percents),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
'VerticalAlignment','bottom')
b(1).FaceColor = 'flat';
b(1).CData(1,:) = plotcolors(1,:);
b(1).CData(2,:) = plotcolors(2,:);
b(1).CData(3,:) = plotcolors(3,:);
ylabel("Num. saccades")
ylim([0 height(saccade_imu_population_table)])
title("Categorization of saccades based on head movement onset time")


%% 3 eye head coupling categories - by file
figure('position',[100         100        1500         800]); 
colororder(plotcolors)
sac_w_hm_ongoing_byfile = file_summary_stats.sacs_with_hm_ongoing;
sac_w_hm_in_window_byfile = file_summary_stats.sacs_with_hm_in_window;
sac_wo_hm_byfile = file_summary_stats.sacs_with_no_hm;
percents = round([file_summary_stats.percent_with_ongoing_hm, file_summary_stats.percent_with_hm_in_window, file_summary_stats.percent_with_no_hm]*100);
b = bar(filenames,[sac_w_hm_ongoing_byfile, sac_w_hm_in_window_byfile, sac_wo_hm_byfile]');
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels1 = strcat(string(percents(:,1)),'%');
labels2 = strcat(string(percents(:,2)),'%');
labels3 = strcat(string(percents(:,3)),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
text(xtips2,ytips2,labels2,'HorizontalAlignment','center','VerticalAlignment','bottom')
text(xtips3,ytips3,labels3,'HorizontalAlignment','center','VerticalAlignment','bottom')
ylabel("Num. saccades")
title("Categorization of saccades based on head movement onset time - by session")


%% 3 eye head coupling categories - by animal
figure('position',[100         100        1500         800]); 
colororder(plotcolors)
sac_w_hm_ongoing_byanimal = animal_summary_stats.sacs_with_hm_ongoing;
sac_w_hm_in_window_byanimal = animal_summary_stats.sacs_with_hm_in_window;
sac_wo_hm_byanimal = animal_summary_stats.sacs_with_no_hm;
percents = round([animal_summary_stats.percent_with_ongoing_hm, animal_summary_stats.percent_with_hm_in_window, animal_summary_stats.percent_with_no_hm]*100);
b = bar(animal_ids,[sac_w_hm_ongoing_byanimal, sac_w_hm_in_window_byanimal, sac_wo_hm_byanimal]');
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels1 = strcat(string(percents(:,1)),'%');
labels2 = strcat(string(percents(:,2)),'%');
labels3 = strcat(string(percents(:,3)),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
text(xtips2,ytips2,labels2,'HorizontalAlignment','center','VerticalAlignment','bottom')
text(xtips3,ytips3,labels3,'HorizontalAlignment','center','VerticalAlignment','bottom')
ylabel("Num. saccades")
ylim([0 height(saccade_imu_population_table)])
title("Categorization of saccades based on head movement onset time - by animal")


%% 3 eye head coupling categories using different saccade thresholds 
figure('position',[200         500        1600         400]); 
thresholds = [0.75, 1, 2, 3, 4, 5];
for i = 1:6
    subplot(1,6,i);
    temp_saccade_imu_population_table = saccade_imu_population_table(saccade_imu_population_table.sac_mag>thresholds(i),:);
    sac_w_hm_ongoing = sum(temp_saccade_imu_population_table.hm_category==-1);
    sac_w_hm_in_window = sum(temp_saccade_imu_population_table.hm_category==1);
    sac_wo_hm = sum(temp_saccade_imu_population_table.hm_category==0);
    percents = round([sac_w_hm_ongoing; sac_w_hm_in_window; sac_wo_hm]./height(temp_saccade_imu_population_table)*100);
    b = bar(["Ongoing HM"; "HM in window"; "No HM in window"],[sac_w_hm_ongoing; sac_w_hm_in_window; sac_wo_hm]);
    set(gca,'TickLabelInterpreter','none')
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = strcat(string(percents),'%');
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
    title(strcat("Sac thresh = ", num2str(thresholds(i)), " deg"))
    ylabel("Num. saccades")
    ylim([0 height(temp_saccade_imu_population_table)])
    b(1).FaceColor = 'flat';
    b(1).CData(1,:) = plotcolors(1,:);
    b(1).CData(2,:) = plotcolors(2,:);
    b(1).CData(3,:) = plotcolors(3,:);
end
sgtitle("Testing different saccade thresholds")


%% 3 eye head coupling categories - licking vs no licking
figure('position',[100         100        1500         800]); 
sac_w_hm_ongoing_licks = sum(saccade_imu_population_table.hm_category==-1 & saccade_imu_population_table.lick_sum>0);
sac_w_hm_in_window_licks = sum(saccade_imu_population_table.hm_category==1 & saccade_imu_population_table.lick_sum>0);
sac_wo_hm_licks = sum(saccade_imu_population_table.hm_category==0 & saccade_imu_population_table.lick_sum>0);
sac_w_hm_ongoing_nolicks = sum(saccade_imu_population_table.hm_category==-1 & saccade_imu_population_table.lick_sum==0);
sac_w_hm_in_window_nolicks = sum(saccade_imu_population_table.hm_category==1 & saccade_imu_population_table.lick_sum==0);
sac_wo_hm_nolicks = sum(saccade_imu_population_table.hm_category==0 & saccade_imu_population_table.lick_sum==0);
sac_licks = sum(saccade_imu_population_table.lick_sum>0);
sac_nolicks = sum(saccade_imu_population_table.lick_sum==0);
percents_licks = round([sac_w_hm_ongoing_licks; sac_w_hm_in_window_licks; sac_wo_hm_licks]/sac_licks*100);
percents_nolicks = round([sac_w_hm_ongoing_nolicks; sac_w_hm_in_window_nolicks; sac_wo_hm_nolicks]/sac_nolicks*100);

subplot(1,2,1)
b = bar(["Ongoing HM"; "HM in window"; "No HM in window"],[sac_w_hm_ongoing_licks; sac_w_hm_in_window_licks; sac_wo_hm_licks]);
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = strcat(string(percents_licks),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
'VerticalAlignment','bottom')
ylabel("Num. saccades")
ylim([0 height(saccade_imu_population_table)])
b(1).FaceColor = 'flat';
b(1).CData(1,:) = plotcolors(1,:);
b(1).CData(2,:) = plotcolors(2,:);
b(1).CData(3,:) = plotcolors(3,:);
title("Saccades with licking")

subplot(1,2,2)
b = bar(["Ongoing HM"; "HM in window"; "No HM in window"],[sac_w_hm_ongoing_nolicks; sac_w_hm_in_window_nolicks; sac_wo_hm_nolicks]);
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = strcat(string(percents_nolicks),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
'VerticalAlignment','bottom')
ylim([0 height(saccade_imu_population_table)])
b(1).FaceColor = 'flat';
b(1).CData(1,:) = plotcolors(1,:);
b(1).CData(2,:) = plotcolors(2,:);
b(1).CData(3,:) = plotcolors(3,:);
title("Saccades with no licking")
sgtitle("Categorization of saccades based on head movement onset time - separated by licking")


%% correlation/distribution of saccade and IMU magnitude
figure('position',[100         100        1500         800]); 
sac_mag = max(saccade_imu_population_table{:,["left_mag","right_mag"]},[],2);
hm_mag = saccade_imu_population_table.max_std_window;
subplot(1,3,1);
scatter(sac_mag,hm_mag,50,'.');
title(strcat("All saccades (corr = ",num2str(corr(sac_mag,hm_mag)),")"))
xlabel("Sac. mag")
ylabel("IMU mag")
ylim([0 ceil(max(hm_mag))])
xlim([0 ceil(max(sac_mag))])
subplot(1,3,2)
histogram(sac_mag)
xlabel("Saccade magnitude")
ylabel("Num saccades")
subplot(1,3,3)
histogram(hm_mag)
xlabel("max IMU magnitude (in window)")
sgtitle("Correlation/distribution of saccade and IMU magnitude")

%% correlation/distribution of saccade and IMU magnitude - by category
figure('position',[100         100        1500         800]); 
sac_mag_ongoing = max(saccades_with_head_movements_ongoing{:,["left_mag","right_mag"]},[],2);
hm_mag_ongoing = saccades_with_head_movements_ongoing.max_std_window;
sac_mag_in_window = max(saccades_with_head_movements_in_window{:,["left_mag","right_mag"]},[],2);
hm_mag_in_window = saccades_with_head_movements_in_window.max_std_window;
sac_mag_wo_hm = max(saccades_without_head_movements{:,["left_mag","right_mag"]},[],2);
hm_mag_wo_hm = saccades_without_head_movements.max_std_window;
subplot(3,3,1);
scatter(sac_mag_ongoing,hm_mag_ongoing,50,'.');
title(strcat("Saccades w/ ongoing HM (corr = ",num2str(corr(sac_mag_ongoing,hm_mag_ongoing)),")"))
xlabel("Sac. mag")
ylabel("IMU mag")
ylim([0 ceil(max(hm_mag))])
xlim([0 ceil(max(sac_mag))])
subplot(3,3,4);
scatter(sac_mag_in_window,hm_mag_in_window,50,'.');
title(strcat("Saccades w/ HM in window (corr = ",num2str(corr(sac_mag_in_window,hm_mag_in_window)),")"))
xlabel("Sac. mag")
ylabel("IMU mag")
ylim([0 ceil(max(hm_mag))])
xlim([0 ceil(max(sac_mag))])
subplot(3,3,7);
scatter(sac_mag_wo_hm,hm_mag_wo_hm,50,'.');
title(strcat("Saccades w/o HM (corr = ",num2str(corr(sac_mag_wo_hm,hm_mag_wo_hm)),")"))
xlabel("Sac. mag")
ylabel("IMU mag")
ylim([0 ceil(max(hm_mag))])
xlim([0 ceil(max(sac_mag))])
subplot(3,3,2)
histogram(sac_mag_ongoing)
xlabel("Saccade magnitude")
ylabel("Num saccades")
subplot(3,3,5)
histogram(sac_mag_in_window)
xlabel("Saccade magnitude")
ylabel("Num saccades")
subplot(3,3,8)
histogram(sac_mag_wo_hm)
xlabel("Saccade magnitude")
ylabel("Num saccades")
subplot(3,3,3)
histogram(hm_mag_ongoing)
xlabel("max IMU magnitude (in window)")
xlim([0 ceil(max(hm_mag))])
subplot(3,3,6)
histogram(hm_mag_in_window)
xlabel("max IMU magnitude (in window)")
xlim([0 ceil(max(hm_mag))])
subplot(3,3,9)
histogram(hm_mag_wo_hm)
xlabel("max IMU magnitude (in window)")
xlim([0 ceil(max(hm_mag))])
sgtitle("Correlation/distribution of saccade and IMU magnitude (by category)")


%% saccade magnitude distribution by category
figure('position',[100         100        1500         800]); 
subplot(1,2,1);hold on;
histogram(sac_mag_ongoing,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability',"EdgeColor","#AA5BA1")
histogram(sac_mag_in_window,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability',"EdgeColor","#2278B5")
histogram(sac_mag_wo_hm,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability',"EdgeColor","#F4793C")
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel("Saccade magnitude")
ylabel("% saccades")
legend("Ongoing HM", "HM in window", "No HM in window")
title("Normalized")
subplot(1,2,2);hold on;
histogram(sac_mag_ongoing,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,"EdgeColor","#AA5BA1")
histogram(sac_mag_in_window,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,"EdgeColor","#2278B5")
histogram(sac_mag_wo_hm,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,"EdgeColor","#F4793C")
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel("Saccade magnitude")
ylabel("Num saccades")
legend("Ongoing HM", "HM in window", "No HM in window")
title("Not normalized")
sgtitle("Saccade magnitude distribution by eye-head coupling category")


%% head movement onset times
figure('position',[100         100        1500         800]); 
histogram(saccade_imu_population_table.onset_ms,50)
title("Onset time of detected head movement")
ylabel("Num saccades")
xlabel("Time since saccade (ms)")


%% correlation between both eyes sac mag, dir
figure('position',[600,500,800,450]);
subplot(1,2,1); hold on;
scatter(saccade_imu_population_table.left_mag,saccade_imu_population_table.right_mag,50,'.')
xlabel("Left eye")
ylabel("Right eye")
lim = max([saccade_imu_population_table.left_mag;saccade_imu_population_table.right_mag]);
xlim([0,lim*1.1])
ylim([0,lim*1.1])
title(strcat("Sac. Mag. (corr = ",num2str(corr(saccade_imu_population_table.left_mag,saccade_imu_population_table.right_mag,Rows="pairwise")),")"))

subplot(1,2,2); hold on;
scatter(saccade_imu_population_table.left_dir,saccade_imu_population_table.right_dir,50,'.')
xlabel("Left eye")
ylabel("Right eye")
xlim([-180 180])
ylim([-180 180])
title(strcat("Sac. Dir. (corr = ",num2str(corr(saccade_imu_population_table.left_dir,saccade_imu_population_table.right_dir,Rows="pairwise")),")"))
sgtitle(strcat('Corr. between saccade mag and dir from both eyes'),"Interpreter","none");


%% angle of saccade for each eye
figure('position',[600,500,800,450]);
subplot(1,2,1);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(rmmissing(saccade_imu_population_table.right_dir)),pi/32:pi/16:pi*65/32);
title("Right eye")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
custom_labels = {'0','30','60','90','120','150','180',...
         '-150','-120','-90','-60','-30'};
pax.ThetaTickLabel = custom_labels;
ax1=gca;

subplot(1,2,2);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(rmmissing(saccade_imu_population_table.left_dir)),pi/32:pi/16:pi*65/32);
title("Left eye")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
pax.ThetaTickLabel = custom_labels;
ax2=gca;
sgtitle(strcat('Saccade directions'),"Interpreter","none");
ax1.RLim(2) = max(ax1.RLim(2),ax2.RLim(2));
ax2.RLim(2) = max(ax1.RLim(2),ax2.RLim(2));
text(-110,-40,"Angle of saccade (degrees)",'Units','pixels')

%% conjugate vs non-conjugate eye movements
temp = saccade_imu_population_table(~isnan(saccade_imu_population_table.left_idx)&~isnan(saccade_imu_population_table.right_idx),:);
conj=temp(abs(wrapTo180(temp.interoccular_angle))<30,:);
nonconj=temp(abs(wrapTo180(temp.interoccular_angle))>=30,:);

figure('position',[100         100        1500         800]); 
b = bar(["Conjugate";"Non-conjugate"],[height(conj);height(nonconj)]);
percents = round([height(conj); height(nonconj)]./height(temp)*100);
set(gca,'TickLabelInterpreter','none')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = strcat(string(percents),'%');
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
'VerticalAlignment','bottom')
ylabel("Num. saccades")
ylim([0 height(temp)])

figure('position',[100         100        1500         800]); 
subplot(2,2,1);
scatter(conj.left_mag,conj.right_mag,50,'.')
xlabel("Left saccade mag")
ylabel("Right saccade mag")
title(strcat("Conjugate eye movements (w/in 30 deg) (corr = ",num2str(corr(conj.left_mag,conj.right_mag)),")"))
xlim([0 10]);
ylim([0 10]);
subplot(2,2,2)
scatter(nonconj.left_mag,nonconj.right_mag,50,'.')
xlabel("Left saccade mag")
ylabel("Right saccade mag")
title(strcat("Non-conjugate eye movements (corr = ",num2str(corr(nonconj.left_mag,nonconj.right_mag)),")"))
xlim([0 10]);
ylim([0 10]);

subplot(2,2,3);
scatter(conj.left_dir,conj.right_dir,50,'.')
xlabel("Left saccade dir")
ylabel("Right saccade dir")
title(strcat("Conjugate eye movements (w/in 30 deg) (corr = ",num2str(corr(conj.left_dir,conj.right_dir)),")"))
xlim([-180 180])
ylim([-180 180])
subplot(2,2,4)
scatter(nonconj.left_dir,nonconj.right_dir,50,'.')
xlabel("Left saccade dir")
ylabel("Right saccade dir")
title(strcat("Non-conjugate eye movements (corr = ",num2str(corr(nonconj.left_dir,nonconj.right_dir)),")"))
xlim([-180 180])
ylim([-180 180])

%% nasal vs temporal saccades
nasal_sac_l = saccade_imu_population_table(saccade_imu_population_table.left_dir<=-45 & saccade_imu_population_table.left_dir>=-135 & saccade_imu_population_table.left_mag>=1,:);
temporal_sac_l = saccade_imu_population_table(saccade_imu_population_table.left_dir>=45 & saccade_imu_population_table.left_dir<=135 & saccade_imu_population_table.left_mag>=1,:);
nasal_sac_r = saccade_imu_population_table(saccade_imu_population_table.right_dir>=45 & saccade_imu_population_table.right_dir<=135 & saccade_imu_population_table.right_mag>=1,:);
temporal_sac_r = saccade_imu_population_table(saccade_imu_population_table.right_dir<=-45 & saccade_imu_population_table.right_dir>=-135 & saccade_imu_population_table.right_mag>=1,:);
nasal_saccades_mag = [nasal_sac_l.left_mag;nasal_sac_r.right_mag];
temporal_saccades_mag = [temporal_sac_l.left_mag;temporal_sac_r.right_mag];

%checking that directions are correct
figure('position',[100         100        1500         800]);
subplot(2,2,1);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(nasal_sac_r.right_dir),pi/32:pi/16:pi*65/32);
title("Right eye- nasal")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
custom_labels = {'0','30','60','90','120','150','180',...
         '-150','-120','-90','-60','-30'};
pax.ThetaTickLabel = custom_labels;
subplot(2,2,2);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(nasal_sac_l.left_dir),pi/32:pi/16:pi*65/32);
title("Left eye- nasal")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
pax.ThetaTickLabel = custom_labels;
subplot(2,2,3);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(temporal_sac_r.right_dir),pi/32:pi/16:pi*65/32);
title("Right eye- temporal")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
pax.ThetaTickLabel = custom_labels;
subplot(2,2,4);
sp1=gca;
pos1 = sp1.Position;
delete(sp1);
pax=polaraxes('Position',pos1);
polarhistogram(pax, deg2rad(temporal_sac_l.left_dir),pi/32:pi/16:pi*65/32);
title("Left eye- temporal")
pax.ThetaZeroLocation='top';
pax.ThetaDir='clockwise';
pax.RLim = [0, max(pax.Children.BinCounts)];
pax.ThetaTick = 0:30:330;
pax.ThetaTickLabel = custom_labels;
sgtitle("Checking that nasal and temporal categorization is correct")


%% nasal vs temporal saccades mag
figure('position',[100         100        1500         800]); hold on;
histogram(nasal_saccades_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
histogram(temporal_saccades_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
legend('nasal','temporal')
xlabel("Saccade mag")
ylabel("% saccades")
title("Magnitude distribution of nasal vs temporal saccades")
    
figure('position',[100         100        1500         800]);
hold on;
histogram(nasal_sac_l.left_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
histogram(nasal_sac_r.right_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
histogram(temporal_sac_l.left_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
histogram(temporal_sac_r.right_mag,0:0.25:10, 'DisplayStyle', 'stairs',"LineWidth",1.5,'Normalization','probability')
legend('nasal L','nasal R','temporal L','temporal R')
xlabel("Saccade mag")
ylabel("% saccades")
sgtitle("Magnitude distribution of nasal vs temporal saccades - split by eye")



FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  saveas(FigHandle, strcat(path, '\fig_',FigName,'.png'));
end

%end