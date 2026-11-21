# Security and private data

iDetectFace is a local application for a supervised check-in station. A person with write access to the data directory can change enrollments or attendance. A photo can produce a match. Use an organizer and a manual alternative when attendance affects someone.

Store the register in a private OS account or access-controlled directory. Choose pseudonymous participant IDs, obtain permission before enrollment and agree on a retention period. Protect backup copies as well as the live files. JPEGs and CSVs are stored locally in plaintext. Linux and macOS create new register files with owner-only access. Windows uses the containing directory's ACL.

Use trusted local files. The evaluation runner accepts relative probe paths inside the manifest directory. Gallery and probe images must be regular JPEG or PNG files of at most 16 MiB and 16 megapixels. Ambiguous equal-distance matches across different identities produce an unknown result.

The current release tree excludes personal images and attendance logs. The public repository's earlier commits contain demonstration portraits and attendance rows. Deleting a file from a later commit does not erase old commits or downloaded clones.

For a suspected vulnerability, use GitHub's private security reporting option if enabled, or contact the maintainer privately through their GitHub profile. Include the affected commit, operating system and observed behavior. Use synthetic identifiers and omit actual portraits, attendance records and credentials.
