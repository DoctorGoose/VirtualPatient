"threads,blas_threads,procs,method,total_time,avg_time_per_sample,avg_num_time_points" | Out-File results.csv

# Define the range of threads, BLAS threads, and processes
$threadsRange = 1..6
$blasThreadsRange = 1..6
$procsRange = 1..6

foreach ($threads in $threadsRange) {
    foreach ($blas_threads in $blasThreadsRange) {
        foreach ($procs in $procsRange) {
            Write-Host "Running benchmark with threads=$threads, blas_threads=$blas_threads, procs=$procs"

            # Set the JULIA_NUM_THREADS environment variable
            $env:JULIA_NUM_THREADS = $threads

            # Run Julia and pass the blas_threads and procs as arguments
            $output = & julia julia_benchmark.jl $blas_threads $procs

            # Parse the results and append to the CSV file
            $methods = @("serial", "pmap", "tmap")
            foreach ($method in $methods) {
                # Extract the line containing the method's results
                $line = $output | Select-String "$method =>"

                if ($line) {
                    # Remove unnecessary characters
                    $data = $line -replace "\s*$method\s*=>\s*\(", "" -replace "\)", ""
                    # Split the data by commas
                    $values = $data -split ","
                    # Initialize variables
                    $total_time = ""
                    $avg_time_per_sample = ""
                    $avg_num_time_points = ""

                    foreach ($val in $values) {
                        $val = $val.Trim()
                        if ($val -like "total_time=*") {
                            $total_time = ($val -split "=")[1].Trim()
                        } elseif ($val -like "avg_time_per_sample=*") {
                            $avg_time_per_sample = ($val -split "=")[1].Trim()
                        } elseif ($val -like "avg_num_time_points=*") {
                            $avg_num_time_points = ($val -split "=")[1].Trim()
                        }
                    }

                    # Append the results to the CSV file
                    "$threads,$blas_threads,$procs,$method,$total_time,$avg_time_per_sample,$avg_num_time_points" | Out-File results.csv -Append
                }
            }

            # Optional: Sleep for a short time to prevent overloading
            Start-Sleep -Seconds 1
        }
    }
}