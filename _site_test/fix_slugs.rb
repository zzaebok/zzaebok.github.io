#!/usr/bin/env ruby

require 'yaml'
require 'fileutils'

# Configuration
NEW_DOMAIN = "https://jaebok-lee.com"
POSTS_DIR = "_posts"

# Process each post
Dir.glob("#{POSTS_DIR}/*.md").each do |post_file|
  content = File.read(post_file)
  
  # Extract front matter
  if content =~ /\A---\n(.*?)\n---(.*)/m
    front_matter_text = $1
    post_content = $2
    
    # Parse frontmatter
    frontmatter = {}
    front_matter_text.split("\n").each do |line|
      if line.include?(':')
        key, value = line.split(':', 2)
        frontmatter[key.strip] = value.strip
      end
    end
    
    # Extract slug from filename and convert underscores to hyphens
    filename = File.basename(post_file, '.md')
    slug = filename.gsub(/^\d{4}-\d{2}-\d{2}-/, '').downcase.gsub('_', '-')
    
    # Update redirect_to URL
    new_url = "#{NEW_DOMAIN}/posts/#{slug}"
    old_url = frontmatter['redirect_to']
    
    if old_url != new_url
      frontmatter['redirect_to'] = new_url
      
      # Reconstruct the file
      new_content = "---\n"
      frontmatter.each do |key, value|
        new_content += "#{key}: #{value}\n"
      end
      new_content += "---#{post_content}"
      
      File.write(post_file, new_content)
      puts "Updated #{File.basename(post_file)}: #{old_url} -> #{new_url}"
    end
  end
end

puts "Slug conversion complete!"