<?php
// Check if the form was submitted
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Save the JSON config to Gui.json
    $config = json_decode($_POST['config'], true);
    file_put_contents('Gui.json', json_encode($config));

    // Run the Python script and handle the progress bar
    $command = 'activate_venv.bat';
    $command = 'python train.py --config Gui.json';
    $handle = popen($command . ' 2>&1', 'r');
    
    while (!feof($handle)) {
        $buffer = fgets($handle);
        echo $buffer . '<br>';
        flush();
        ob_flush();
        usleep(100000); // Adjust this value to control the progress bar update frequency
    }
    
    pclose($handle);
}
?>
