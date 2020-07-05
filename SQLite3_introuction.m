%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script shows how to use SQLite3 in Matlab.                         %
% The example is adopted from https://www.sqlite.org/foreignkeys.html.    %
% Developed by Taeyong Kim from the Seoul National University             %
% July 5, 2020 chs5566@snu.ac.kr                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;

% Basic parameters
Name_db = 'Example_db.db'; % Name of database for saving hysteresis

% Make a database --> SQLite3
dbfile = fullfile(pwd,Name_db);  % Specify the file name in the current working folder.
con = sqlite(dbfile,'create');   % connect to the database

% Generate table
art_sql = ['create table artist (artistid integer PRIMARY KEY, ' ...
                                'artistname text NOT NULL, ' ...
                                '"Nationality(optional)" text)'];
exec(con, art_sql)

tra_sql = ['create table track (trackid integer, ' ...
                               'trackname text, ' ...
                               'trackartist INTEGER, ' ...
                               'FOREIGN KEY (trackartist) REFERENCES artist (artistid))'];
exec(con, tra_sql)

% Store the datasets to the database
art_val_sql = {'artistid','artistname','"Nationality(optional)"'};
insert(con, 'artist', art_val_sql, ...
       {1, 'Dean Martin', 'USA'});
insert(con, 'artist', art_val_sql, ...
       {2, 'Frank Sinatra', 'Canada'});

tra_val_sql = {'trackid','trackname','trackartist'};
insert(con, 'track', tra_val_sql, ...
       {11, 'That''s Amore', 1});
insert(con, 'track', tra_val_sql, ...
       {12, 'Christmas Blues', 1});
insert(con, 'track', tra_val_sql, ...
       {13, 'My Way', 2});   
   
% Fetch dataset from the database
fetch_artist = 'SELECT * FROM artist';
artist_values = fetch(con,fetch_artist);

fetch_track = 'SELECT * FROM track WHERE trackartist==1';
track_values = fetch(con,fetch_track);

close(con)   


