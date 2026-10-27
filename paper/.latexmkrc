# Resolve the vendored conference style and bibliography style in editor builds.
use Cwd qw(abs_path);

my $vendor_dir = abs_path('vendor');
for my $search_path ('TEXINPUTS', 'BSTINPUTS') {
    $ENV{$search_path} = $vendor_dir . ':' . ($ENV{$search_path} // '');
}
