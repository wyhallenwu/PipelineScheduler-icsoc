#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long;
use File::Path qw(make_path);
use POSIX qw(strftime);


sub usage {
    print "Usage: keepn.pl [--lines N]\n";
    print "       --lines N  Number of lines to keep in memory (default: 100)\n";
    exit;
}
my $keep = 100;
GetOptions('lines=i' => \$keep) or usage();

$0 = 'outputbuffer';
my $okay = 1;
foreach my $sig (qw(HUP INT PIPE TERM USR1)) {
    $SIG{$sig} = sub { $okay = 0 };
}

# Read lines from input
my @buf;
while ($okay and defined(my $line = <STDIN>)) {
    push @buf, $line;
    if (@buf > $keep) {
        shift @buf;
    }
}

# Write the buffered lines to the log file
my $log_dir = '../logs';
make_path($log_dir) unless -d $log_dir;
my $timestamp = strftime("%Y%m%d%H%M%S", localtime);
my $log_file = "$log_dir/error_$timestamp.log";

open my $fh, '>', $log_file or die "Cannot open $log_file: $!";
print $fh $_ for @buf;
close $fh or warn "Cannot close $log_file: $!";
print "Log written to $log_file\n";
