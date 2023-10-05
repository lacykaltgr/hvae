#!/bin/bash

# Check if this is the first run
if [ ! -e /firstrun ]; then

    echo "First run"

    # Create a root password
    #passwd

    # Set the timezone
    #dpkg-reconfigure tzdata

    # Create the group for files on the fileserver
    #groupadd -g 1000 fileserv

    # Create your LDAP user as a Docker-local user
    #useradd -d /home/laszlofreund -g fileserv -G fileserv,sudo -m -N -u 420.laszlofreund

    # Set a password for your new local user
    #passwd laszlofreund

    # Test your new user's ability to write files on the fileserver
    #su - username -c "touch /mnt/foo"
    #su - username -c "ls -l /mnt"
    #su - username -c "rm /mnt/foo"

    # Mark that it's not the first run anymore
    #touch /firstrun
fi

# Start your Docker container
exec "$@"