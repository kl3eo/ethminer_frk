#!/usr/bin/perl
my $tainted = 0;

my $wget_cmd = '/bin/ps aux';

open (IN, '-|', $wget_cmd)
	or die ("Can't ps");

   while (!eof(IN)) {
	my $q = readline (*IN);
#print $q."\n";		

	if ($q =~ / ethminer /) {
		$tainted = 1;
	}
   }
close (IN);


unless ($tainted) {

my $wget_cmd = "cd /root && ./ethmi";

system($wget_cmd) if (-f "/home/op/home/op/ethmi_started");
system('date >> /root/ethmi_resets');
};

exit;

